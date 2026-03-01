import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from flux import FluxPipeline
from transformers import CLIPProcessor, CLIPModel
from utils import steering_txt_data, apply_txt_steering

def retrieve_latents(
    encoder_output: torch.Tensor, generator: torch.Generator | None = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


# --- 1. Dataset: Loads Image + Matching .txt Prompt ---
class FluxImageDataset(Dataset):
    def __init__(self, img_dir, size=1024):
        self.img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
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
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        
        # Load prompt from same filename .txt
        
        prompt = self.prompts[idx]
        
        return {'image': self.transform(img), 'prompt': prompt}

# --- 2. Multi-Layer Steering Optimizer ---
class MultiLayerSteeringOptimizer:
    def __init__(self, flux_pipe, target_concept, log_dir, device="cuda", size=1024, path=None):
        self.device = device
        self.pipe = flux_pipe
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # CLIP Setup
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        with torch.no_grad():
            inputs = self.clip_processor(text=[target_concept], return_tensors="pt", padding=True).to(device)
            self.target_features = self.clip_model.get_text_features(**inputs)
            self.target_features /= self.target_features.norm(dim=-1, keepdim=True)
        
        self.size = size
        
        if path is not None:
            self.anchor_vector = torch.load(path, weights_only=False)
        else:
            self.anchor_vector = None

    def get_x0_prediction(self, x_t, t, steering_params, prompt_embeds):
        handles = []
        
        # We hook the first 19 blocks (Double Blocks)
        def create_hook(layer_idx):
            def hook_fn(module, input, output):
                if len(output) != 2: return output
                # output[0] is the hidden_state for Flux Double Blocks
                vec = steering_params[f"layer_{layer_idx}"]
                vec = vec / (vec.norm() + 1e-6)
                orig_norm = torch.norm(output[1], dim=-1, keepdim=True) + 1e-6
                modified_h = output[1] + vec.to(output[1].device, output[1].dtype)
                modified_h = modified_h / (torch.norm(modified_h, dim=-1, keepdim=True) + 1e-6)
                modified_h = (modified_h * orig_norm).to(output[1].dtype)
                return (output[0], modified_h)
            return hook_fn

        layer_id = 0
        for name, module in self.pipe.transformer.named_modules():
            if name.endswith("attn"):
                handles.append(module.register_forward_hook(create_hook(layer_id)))
               
                layer_id += 1
                
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
                guidance=guidance,
                encoder_hidden_states=prompt_embeds["encoder_hidden_states"],
                pooled_projections=prompt_embeds["pooled_projections"],
                return_dict=False,
                txt_ids=text_ids,
                img_ids=latent_image_ids

            )[0]
        finally:
            for h in handles: h.remove()
        
        t_norm = t / 1000.0
        return x_t - t_norm.to(x_t.dtype) * model_output

    def run_optimization(self, dataset, args):
        # Initialize 19 parameters: [1, 512, 3072]
        # steering_params = nn.ParameterDict({
        #     f"layer_{i}": nn.Parameter(torch.zeros((1, 512, 3072), device=self.device, dtype=torch.bfloat16))
        #     for i in range(19)
        # })
        txt_steering = {'vector': torch.load('experiments/flux_schnell/add/final_steering/glasses/base_0.85_20_text_diff.pt'), 'strength': 2.5}
        
        
        steering_params = nn.ParameterDict()
        for i in range(19):
            layer_key = f"layer_{i}"
            if self.anchor_vector and layer_key in self.anchor_vector[0]:
                # Load from anchor and convert to parameter
                init_val = self.anchor_vector[0][layer_key].to(self.device, dtype=torch.bfloat16)
            else:
                # Fallback to zeros if layer missing
                init_val = torch.zeros((1, 512, 3072), device=self.device, dtype=torch.bfloat16)
            
            steering_params[layer_key] = nn.Parameter(init_val)
        
        optimizer = torch.optim.Adam(steering_params.values(), lr=args.lr)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        
        for i in range(args.iters):
            optimizer.zero_grad()
            mean_loss_concept = []
            mean_loss_dist = []
            for j, sample in enumerate(loader):
                batch_imgs = sample['image']
                batch_prompts = sample['prompt']
                
                
                # Encode Latents & Prompts
                
                latents = retrieve_latents(self.pipe.vae.encode(batch_imgs.to(self.device, dtype=torch.bfloat16)))
                latents = (latents - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
                width, height = latents.shape[3], latents.shape[2]
                latents = self.pipe._pack_latents(latents, latents.shape[0], latents.shape[1], latents.shape[2], latents.shape[3])

                with torch.no_grad():
                    out = self.pipe.encode_prompt(prompt=batch_prompts)
                    # p_embeds = {"encoder_hidden_states": out[0], "pooled_projections": out[1]}
                    if txt_steering['vector'] is not None:
                        pooled_style, seqs_style = steering_txt_data(txt_steering['vector'], txt_steering['strength'], out[0], mean=False, ssim=True, pooled=True, normed=False, num=None)
                        #print(pooled_style.shape, seqs_style.shape)
                    else:
                        assert False
                    new_pooled_embeds, new_prompt_embeds = apply_txt_steering( out[1],  out[0], pooled_style, seqs_style, normed=False)
                    p_embeds = {"encoder_hidden_states": new_prompt_embeds, "pooled_projections": new_pooled_embeds}
                
                # Flow Matching Noise
                t = torch.randint(700, 950, (1,), device=latents.device).float()
                t_norm = t / 1000.0
                noise = torch.randn_like(latents)
                x_t = (1 - t_norm) * latents + t_norm * noise
                
                # Steer & Predict
                pred_x0_latent = self.get_x0_prediction(x_t, t, steering_params, p_embeds)
                
                # Decode & Semantic Loss
                latents = self.pipe._unpack_latents(pred_x0_latent, height*8, width*8, self.pipe.vae_scale_factor)
                latents = (latents / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
                pix = self.pipe.vae.decode(latents, return_dict=False)[0]
                #image = self.pipe.image_processor.postprocess(pix, output_type='pil')

                pix_normalized = (pix + 1.0) / 2.0
                pix_resized = F.interpolate(pix_normalized, size=(224, 224), mode='bilinear')
                
                img_feats = self.clip_model.get_image_features(pix_resized.to(self.clip_model.device))
                img_feats = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-6)
                
                loss_concept = 1 - torch.cosine_similarity(img_feats, self.target_features).mean()
                # loss_reg = sum(torch.norm(v, p=2) for v in steering_params.values()) * args.reg_weight
                loss_concept = loss_concept 
                # DISTANCE LOSS (don't go far from anchor)
                loss_dist = 0
                if self.anchor_vector:
                    for name, param in steering_params.items():
                        anchor = self.anchor_vector[0][name].to(self.device, dtype=torch.bfloat16)
                        loss_dist += F.mse_loss(param, anchor)
                
                total_loss = loss_concept  + (loss_dist * args.dist_weight)
                
                (total_loss).backward()
                
                
                if i % 10 == 0:
                    image = self.pipe.image_processor.postprocess(pix.detach(), output_type='pil')
                    os.makedirs('test_optimization', exist_ok=True)
                    image[0].save(f'test_optimization/{i}_{j}.png')
                    
                mean_loss_concept.append(loss_concept.item())
                mean_loss_dist.append(loss_dist.item())
            print(f"Step {i} | Loss: {np.mean(mean_loss_concept):.4f}, {np.mean(mean_loss_dist):.4f},  {(loss_dist.item()* args.dist_weight):.4f}")
            optimizer.step()
        # Return in specific dictionary format
        return {"0": {name: param.detach().cpu() for name, param in steering_params.items()}}

# --- 3. Refactored Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--init_path", type=str, default=None)
    parser.add_argument("--target_concept", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.1-schnell")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--dist_weight", type=float, default=0.02)
    parser.add_argument("--output_path", type=str, default="steering_vector.pt")
    parser.add_argument("--log_dir", type=str, default="logs/flux_steer")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {args.model_id}...")
    pipe = FluxPipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, device_map="balanced",  use_safetensors=True)
    
    dataset = FluxImageDataset(args.data_dir, size=args.size)
    opt_engine = MultiLayerSteeringOptimizer(pipe, args.target_concept, args.log_dir, device, size=args.size, path=args.init_path)

    # Note: We pass the dataset directly because prompts change per image
    final_vec_dict = opt_engine.run_optimization(dataset, args)

    torch.save(final_vec_dict, args.output_path)
    print(f"Success! 19 layers saved to {args.output_path}")

if __name__ == "__main__":
    main()