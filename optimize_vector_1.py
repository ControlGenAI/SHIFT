import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from diffusers import FluxPipeline 

# --- Утилиты ---
def retrieve_latents(encoder_output, generator=None, sample_mode="sample"):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

class FluxPairedDataset(Dataset):
    def __init__(self, source_dir, target_dir, size=512):
        self.source_paths = sorted([os.path.join(source_dir, f) for f in os.listdir(source_dir) 
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.target_paths = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir) 
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.CenterCrop((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        self.txt_path = 'prompts_collection/dataset_creation/dataset_prompts_add.txt'
        self.prompts = []
        if os.path.exists(self.txt_path):
            with open(self.txt_path, "r") as f:
                self.prompts = [line.strip() for line in f if line.strip()]


    def __len__(self):
        return min(len(self.source_paths), len(self.target_paths))

    def __getitem__(self, idx):
        src_img = Image.open(self.source_paths[idx]).convert("RGB")
        tgt_img = Image.open(self.target_paths[idx]).convert("RGB")
        prompt = self.prompts[idx]
        return {
            'source': self.transform(src_img),
            'target': self.transform(tgt_img),
            'prompt': prompt
        }

# --- Optimizer ---
class FeatureMatchingOptimizer:
    def __init__(self, flux_pipe, log_dir, target_concept, device="cuda", size=512, path=None):
        self.device = device
        self.pipe = flux_pipe
        self.size = size
        self.target_concept = target_concept
        
        if path is not None:
            self.anchor_vector = torch.load(path, weights_only=False)
        else:
            self.anchor_vector = None
        self.target_activations = {}
        self.writer = SummaryWriter(log_dir=log_dir)

    def encode_vae(self, img):
        latents = retrieve_latents(self.pipe.vae.encode(img.to(self.device, dtype=torch.bfloat16)))
        latents = (latents - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
        latents = self.pipe._pack_latents(latents, latents.shape[0], latents.shape[1], latents.shape[2], latents.shape[3])
        return latents

    def get_x0_prediction(self, x_t, t, steering_params, prompt_embeds, is_target_pass=True):
        handles = []
        
        def create_hook(layer_idx):
            def hook_fn(module, input, output):
                if len(output) != 2: return output
                if is_target_pass:
                    # Сохраняем таргет (отвязываем от графа)
                    self.target_activations.append(output[1].detach())
                    return output
                else:
                    vec = steering_params[f"layer_{layer_idx}"]
                    # Нормализация вектора для стабильности
                    vec = vec / (vec.norm() + 1e-6)
                    orig_norm = torch.norm(output[1], dim=-1, keepdim=True) + 1e-6
                    
                    # Применяем стиринг
                    modified_h = output[1] + vec.to(output[1].device, output[1].dtype)
                    
                    # Возвращаем к оригинальной норме (опционально, но помогает не "взрывать" модель)
                    modified_h = modified_h / (torch.norm(modified_h, dim=-1, keepdim=True) + 1e-6)
                    modified_h = (modified_h * orig_norm).to(output[1].dtype)
                    
                    self.input_activations.append(modified_h)
                    return (output[0], modified_h)
                
            return hook_fn

        layer_id = 0
        for name, module in self.pipe.transformer.named_modules():
            if name.endswith("attn"):
                handles.append(module.register_forward_hook(create_hook(layer_id)))
               
                layer_id += 1
                
        # Подготовка параметров для Flux
        text_ids = torch.zeros(prompt_embeds["encoder_hidden_states"].shape[1], 3).to(device=self.pipe.device, dtype=self.pipe.dtype)
        guidance = torch.full([1], 0.0, device=self.pipe.device, dtype=torch.float32)
        guidance = guidance.expand(x_t.shape[0])
        guidance = None
        latent_image_ids = self.pipe._prepare_latent_image_ids(x_t.shape[0], self.size//16, self.size//16, self.pipe.device, self.pipe.dtype)
        x_t = x_t.to(self.pipe.dtype)
        try:    
            model_output = self.pipe.transformer(
                hidden_states=x_t,
                timestep=t.expand(x_t.shape[0]) / 1000,
                encoder_hidden_states=prompt_embeds["encoder_hidden_states"].to(self.pipe.dtype),
                pooled_projections=prompt_embeds["pooled_projections"].to(self.pipe.dtype),
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                return_dict=False
            )[0]
        finally:
            for h in handles: h.remove()
        
        t_norm = t / 1000.0
        return x_t - t_norm.to(x_t.dtype) * model_output


    def run_optimization(self, dataset, args):
        steering_params = nn.ParameterDict()
        for i in range(19):
            layer_key = f"layer_{i}"
            if self.anchor_vector and layer_key in self.anchor_vector[0]:
                init_val = self.anchor_vector[0][layer_key].to(self.device, dtype=torch.bfloat16)
            else:
                init_val = torch.zeros((1, 512, 3072), device=self.device, dtype=torch.bfloat16)
            
            steering_params[layer_key] = nn.Parameter(init_val)

        optimizer = torch.optim.Adam(steering_params.values(), lr=args.lr)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        for i in range(args.iters):
            mean_feat_loss, mean_loss_dist = [], []
            
            for j, sample in tqdm(enumerate(loader), desc=f"Epoch {i}", total=len(loader)):
                optimizer.zero_grad()
                src_imgs, tgt_imgs = sample['source'].to(self.device), sample['target'].to(self.device)
                t = torch.tensor([args.timestep], device=self.device).float()
                t_norm = t / 1000.0

                with torch.no_grad():
                    src_lat = self.encode_vae(src_imgs)
                    tgt_lat = self.encode_vae(tgt_imgs)
                    noise = torch.randn_like(src_lat)
                    x_t_src = (1 - t_norm) * src_lat + t_norm * noise
                    x_t_tgt = (1 - t_norm) * tgt_lat + t_norm * noise
                    
                    # Промпты: обычный и с концептом
                    out = self.pipe.encode_prompt(prompt=sample['prompt'])
                    p_embeds = {"encoder_hidden_states": out[0], "pooled_projections": out[1]}
                    
                    out_tgt = self.pipe.encode_prompt(prompt=[p + " " + self.target_concept for p in sample['prompt']])
                    p_embeds_target = {"encoder_hidden_states": out_tgt[0], "pooled_projections": out_tgt[1]}

                # 1. Target Pass (собираем эталонные активации)
                self.target_activations = []
                with torch.no_grad():
                    pred_x0_target = self.get_x0_prediction(x_t_tgt, t, steering_params, p_embeds_target, is_target_pass=True)
                
                # 2. Steer Pass (собираем модифицированные активации)
                self.input_activations = []
                pred_x0_latent = self.get_x0_prediction(x_t_src, t, steering_params, p_embeds, is_target_pass=False)

                # 3. Расчет Loss
                current_feat_loss = 0
                for in_act, tgt_act in zip(self.input_activations, self.target_activations):
                    if args.type_loss == 'mse':
                        current_feat_loss += F.mse_loss(in_act, tgt_act)
                    else: # cos
                        # Cosine similarity loss (1 - sim)
                        cos_sim = F.cosine_similarity(in_act, tgt_act, dim=-1).mean()
                        current_feat_loss += (1 - cos_sim)

                loss_dist = sum(torch.norm(v) for v in steering_params.values())
                total_loss = current_feat_loss + (loss_dist * args.dist_weight)
                total_loss.backward()
                optimizer.step()
                
                mean_feat_loss.append(current_feat_loss.item())
                mean_loss_dist.append(loss_dist.item())

                # Визуализация
                if i % 10 == 0:
                    self.save_debug_image(pred_x0_latent, f"{i}_{j}.png")
                    self.save_debug_image(pred_x0_target, f"{i}_{j}_target.png")

            print(f"Epoch {i} | Feat Loss: {np.mean(mean_feat_loss):.4f} | Dist Loss: {np.mean(mean_loss_dist)*args.dist_weight:.4f}")
        
        return {"0": {name: param.detach().cpu() for name, param in steering_params.items()}}

    def save_debug_image(self, latents, name):
        os.makedirs('test_optimization_1', exist_ok=True)
        h, w = self.size, self.size
        latents = self.pipe._unpack_latents(latents, h, w, self.pipe.vae_scale_factor)
        latents = (latents / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
        with torch.no_grad():
            pix = self.pipe.vae.decode(latents, return_dict=False)[0]
        img = self.pipe.image_processor.postprocess(pix.detach(), output_type='pil')[0]
        img.save(f'test_optimization_1/{name}')

# --- Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, required=True)
    parser.add_argument("--target_dir", type=str, required=True)
    parser.add_argument("--target_concept", type=str, default="wearing glasses")
    parser.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.1-schnell")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--timestep", type=float, default=300.0)
    parser.add_argument("--dist_weight", type=float, default=1e-4)
    parser.add_argument("--type_loss", type=str, choices=['mse', 'cos'], default='cos')
    parser.add_argument("--output_path", type=str, default="feature_matching_vector.pt")
    parser.add_argument("--init_path", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default="logs/flux_steer")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = FluxPipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(device)
    
    dataset = FluxPairedDataset(args.source_dir, args.target_dir, size=512)
    opt = FeatureMatchingOptimizer(pipe, args.log_dir, args.target_concept, device, size=512, path=args.init_path)

    final_vector = opt.run_optimization(dataset, args)
    torch.save(final_vector, args.output_path)

if __name__ == "__main__":
    main()