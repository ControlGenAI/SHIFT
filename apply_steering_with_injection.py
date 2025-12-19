import torch
import numpy as np
import os
import torch.nn.functional as F
import argparse
import math
from tqdm import tqdm
from typing import Dict, Any, Optional, List, Callable, Tuple, Union

from PIL import Image
import torchvision.transforms as T
# Assuming these are custom/external modules that should remain
from sd_3 import StableDiffusion3Pipeline 
from flux import FluxPipeline 
#from diffusers import StableDiffusion3Pipeline 
from sd_3_injection import JointAttnProcessor2_Injection 
import sklearn.svm._classes 
from utils import load_steering_data, calculate_cls_score, norm_based_steering_f, orthogonal_projection_steering


# ==============================================================================
# ------------------------------ Main Steering Logic ---------------------------
# ==============================================================================

def apply_attention_steering(
    pipe: StableDiffusion3Pipeline,
    svm_model_path: Optional[str] = None,
    scores_path: Optional[str] = None,
    token_best_path: Optional[str] = None,
    steering_vectors: Dict[int, Dict[str, Union[torch.Tensor, List[torch.Tensor]]]] = None,
    strength: float = 1.0,
    block: Union[str, List[int]] = 'all',
    t_structure: int = 0,
    t_steering=0,
    block_structure: int = 30,
    activations: str = 'attn_enc',
    task: str = 'add concept',
    cls_min: float = 3.0,
    iterative_refinement: bool = False,
    orthogonal_projection: bool = False,
    norm_based_steering: bool = False,
    quantile_type: str = 'block', # 'block', 'timestep', or 'no'
    quantile_level: float = 0.5,
    cls_type: str = 'steep',
    model_type=None,
) -> Tuple[Callable, Callable]:
    """
    Applies attention steering vectors during generation by registering forward hooks.

    Returns:
        A tuple: (step_callback, remove_hooks)
    """
    if steering_vectors is None:
        raise ValueError("Steering vectors must be provided.")

    hook_handles = []
    current_step = 0
    
    # 1. Load Data
    models, tokens_best, scores_all, quantiles = load_steering_data(
        svm_model_path, scores_path, token_best_path, quantile_level
    )
    
    if quantile_type == 'no':
        scores_all = None
    if isinstance(block, str) and block.lower() == 'all':
        steering_blocks = 'all'
    elif isinstance(block, list):
        steering_blocks = set(block)
    else:
        # Assume block is an int list or compatible format
        steering_blocks = set(block)
        

    def step_callback(step_idx: int, timestep: int, latents: torch.Tensor, **kwargs):
        """Callback executed before each main step of the diffusion process."""
        nonlocal current_step
        current_step = step_idx
        
    def steering_hook(layer_idx: int) -> Callable:
        """Creates the forward hook function for a specific layer."""
        def hook(module, input: Tuple[torch.Tensor], output: Tuple[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
            nonlocal current_step
            device = output[0].device
            dtype = output[0].dtype

            
            # --- Check Conditions for Steering ---
            if current_step not in steering_vectors or f"layer_{layer_idx}" not in steering_vectors[current_step]:
                return output
            
            # Check if layer is in the specified steering blocks
            is_in_block = (steering_blocks == 'all' or layer_idx in steering_blocks)

            if not is_in_block:
                if 'flux' in model_type.lower():
                    layers_idx_tg = 56
                else: 
                    layers_idx_tg = 23
                
                if layer_idx == layers_idx_tg:
                    current_step += 1
                return output
            
            is_in_t = (t_steering == 'all' or current_step in t_steering)

            if not is_in_t:
                return output

            # Load steering data for the current step and layer
            vec_data = steering_vectors[current_step][f"layer_{layer_idx}"]
            layer_models = models[current_step][f"layer_{layer_idx}"] if models is not None else None
            
            # Handle single vs. multiple SVM models (if layer_models is a list/tuple)
            if not isinstance(layer_models, list) and not isinstance(layer_models, tuple):
                layer_models = [layer_models] if layer_models is not None else []
            
            if len(output) == 2:
                output_idx = 1 if activations == 'attn_enc' else 0 
                activations_tuple = output
                double_block = True
                if 'flux' in model_type.lower():
                    activations_to_modify = output[output_idx][0].clone()
                else:
                    activations_to_modify = output[output_idx][1].clone() # Shape [L, D] (for tokens)
            else:
                if layer_idx < 30:
                    if 'flux' in model_type.lower():
                        layers_idx_tg = 56
                    else: 
                        layers_idx_tg = 23
                    
                    if layer_idx == layers_idx_tg:
                        current_step += 1
                    return output
                else:
                    double_block = False
                    activations_to_modify = output[0, :512, :].clone()

            original_norm = torch.norm(activations_to_modify.float(), dim=-1, keepdim=True) + 1e-6
            normalized_activations = activations_to_modify.clone() / original_norm # Clone to modify in place

            # Check if there are best tokens or if we steer all tokens
            token_indices = tokens_best[current_step][layer_idx] if tokens_best is not None else None
            steer_all_tokens = token_indices is None or len(token_indices) == 0

            # Convert steering vector to correct device/dtype
            if isinstance(vec_data, list):
                # Handle list of vectors (e.g., from an ensemble or multi-part steering)
                steering_tensor = torch.stack(vec_data, dim=0).to(device=device, dtype=dtype) # [N, D]
            else:
                steering_tensor = vec_data.to(device=device, dtype=dtype).squeeze(0) # [D] or [L, D]

            if activations == 'attn_enc' or activations == 'attn_im':
                if task == 'remove':
                    #assert False
                    v_norm = torch.norm(steering_tensor.float(), dim=-1, keepdim=True)
                    steering_tensor = steering_tensor / (v_norm + 1e-6)

                    #sim = (normalized_activations * steering_tensor[0]).sum(dim=1, keepdim=True)
                    sim = (normalized_activations * steering_tensor).sum(dim=1, keepdim=True)
    
                    sim = F.relu(sim)  # Shape (N, 1)

                    score_value = torch.tensor([1.0]).unsqueeze(1).to(device, dtype).repeat(steering_tensor.shape[0], 1)
                    
                    score_mask = 1.0  
                    if layer_models:
                        if steer_all_tokens:
                            a = normalized_activations.mean(0)[None].cpu().clone() 
                        else:
                            a = normalized_activations[token_indices].mean(0)[None].cpu().clone() 
                        
                        
                        scoreeee_all = []
                        distance_all = []
                        score_cls_all = []
                        # Iterate through all SVM models for this layer/step (if it's an ensemble)
                        for model_instance in layer_models:
                            scoreeee, score_cls, distance = calculate_cls_score(a, cls_min, model=model_instance, cls_type=cls_type, use_distance=False, task=task)
                            scoreeee_all.append(scoreeee)
                            distance_all.append(distance)
                            score_cls_all.append(score_cls) 
                        

                        #score_value = torch.mean(torch.tensor(scoreeee_list)).to(device, dtype)
                        if 'flux' in model_type.lower():
                            score_value = torch.tensor(scoreeee_all).unsqueeze(1).to(device, dtype)
                            score_value = torch.ones_like(score_value)
                        else:
                            score_value = torch.tensor(scoreeee_all).unsqueeze(1).to(device, dtype)
                            # score_value = torch.ones_like(score_value)
                            #print(layer_idx, current_step, score_value, scoreeee, score_cls, cls_min, distance_all) 
                            #print(layer_idx)
                        
                    #subtraction_term = strength * sim * (steering_tensor * score_value)[0]
                    subtraction_term = strength * sim * (steering_tensor * score_value[0])
                    # 5. Perform the subtraction
                    steered_normalized_output = activations_to_modify - subtraction_term

                    vector_direction = steered_normalized_output.float()
                    vector_direction = vector_direction / (torch.norm(vector_direction, dim=-1, keepdim=True) + 1e-6)
                    print(current_step,score_value, score_cls_all, layer_idx, calculate_cls_score(vector_direction.cpu().float().mean(0, keepdim=True), cls_min, model=layer_models[0], cls_type=cls_type, use_distance=False, task='remove'))
                    # Step B: Multiply by the original norm to restore magnitude
                    vector_restored_norm = vector_direction * original_norm.float()
                    
                    # Step C: Assemble the final output tuple
                    new_output_embeddings = vector_restored_norm.to(dtype)
                    

                    if double_block:
                        if 'flux' in model_type.lower():
                            activations_tuple[output_idx][0] = new_output_embeddings
                        else:
                            activations_tuple[output_idx][1] = new_output_embeddings

                        new_output = (
                            activations_tuple[0], # Keep QKV output untouched
                            activations_tuple[1], # Use the norm-restored embeddings
                        )
                    else:
                        output[0, :512] = new_output_embeddings
                        new_output = output
                    
                    if 'flux' in model_type.lower():
                        layers_idx_tg = 56
                    else: 
                        layers_idx_tg = 23
                    
                    if layer_idx == layers_idx_tg:
                        current_step += 1
                    
                    return new_output

                ################################################
                if task == 'add concept':
                    v_norm = torch.norm(steering_tensor.float(), dim=-1, keepdim=True)
                    steering_tensor = steering_tensor / (v_norm + 1e-6)

                # 1. Calculate Score (based on SVM distance or fallback)
                score_value = torch.tensor([1.0]).unsqueeze(1).to(device, dtype).repeat(steering_tensor.shape[0], 1)
                score_mask = 1.0  
                if layer_models:
                    if steer_all_tokens:
                        a = normalized_activations.mean(0)[None].cpu().clone() 
                    else:
                        a = normalized_activations[token_indices].mean(0)[None].cpu().clone() 
                    
                    
                    scoreeee_all = []
                    distance_all = []
                    score_cls_all = []
                    # Iterate through all SVM models for this layer/step (if it's an ensemble)
                    for model_instance in layer_models:
                        scoreeee, score_cls, distance = calculate_cls_score(a, cls_min, model=model_instance, cls_type=cls_type, use_distance=False)
                        scoreeee_all.append(scoreeee)
                        distance_all.append(distance)
                        score_cls_all.append(score_cls) 
                    

                    #score_value = torch.mean(torch.tensor(scoreeee_list)).to(device, dtype)
                    if 'flux' in model_type.lower():
                        score_value = torch.tensor(scoreeee_all).unsqueeze(1).to(device, dtype)
                        score_value = torch.ones_like(score_value)
                    else:
                        score_value = torch.tensor(scoreeee_all).unsqueeze(1).to(device, dtype)
                        
                # 2. Apply Quantile Mask (for best blocks/tokens)
                if scores_all is not None:
                    # Determine the quantile threshold for the current step/layer
                    q = quantiles
                    if quantile_type == 'block':
                        q = quantiles[layer_idx]
                    elif quantile_type == 'timestep':
                        q = quantiles[current_step]
                   
                    score_mask = (scores_all[current_step][layer_idx] >= q).float().to(device)


                # 3. Calculate Scaling and Steering Vector
                adjustment_scale = strength * score_value * score_mask # Base scaling factor
                if not double_block:
                    adjustment_scale /= 2

                
                if norm_based_steering:
                    # Scale based on inverse token norm
                    norm_based_scaling = norm_based_steering_f(activations_to_modify.clone()).to(dtype)
                    adjustment_scale *= norm_based_scaling # [L, 1] tensor

                steering_to_add = steering_tensor.to(dtype)
                
                if orthogonal_projection:
                    # Calculate projection orthogonal to the current normalized output
                    steering_to_add = orthogonal_projection_steering(activations_to_modify.clone(), steering_tensor, normalize=True).to(dtype)


                if iterative_refinement:
                
                    max_iter = 10
                    dist_threshold_pullback = 3.0
                    dist_threshold_push = 2.0
                    damping_factor_init = 0.3
                    
                    # Start by applying the initial step, scaled by the total adjustment
                    # We apply the initial large step to the normalized tensor
                    steered_normalized_output = normalized_activations + steering_to_add * adjustment_scale
                    
                    temp_normalized_activations = steered_normalized_output.clone()
                    
                    for iter_idx in tqdm(range(max_iter), desc=f"Refine Step {current_step}/{layer_idx}"):
                        
                        # Recalculate mean activation for SVM check, using the norm-restored version
                        # Must restore norm temporarily to get a meaningful mean activation for the SVM model,
                        # which was trained on full activations or normalized ones, but checking against *current* state.
                        temp_full_activations = temp_normalized_activations * original_norm
                        a_iter = temp_full_activations.mean(0)[None].cpu().clone()
                        a_iter = a_iter / (a_iter.norm(dim=-1) + 1e-6) # Normalize mean direction
                        
                        if not layer_models: break
                        model_instance = layer_models[0]
                        
                        # Calculate SVM feedback
                        # NOTE: Assuming pred_class 1 is the target concept.
                        pred_class = model_instance.predict(a_iter)[0] 
                        distance = model_instance.decision_function(a_iter)[0]
                        
                        current_damping = damping_factor_init * (0.5 ** iter_idx) 
                        
                        # Define the *unit direction* to push/pull by, applying orthogonal rule if requested
                        if orthogonal_projection:
                            # Ensure current_steering is orthogonal to the *current* state a_iter
                            current_steering = orthogonal_projection_steering(
                                a_iter.to(device), steering_tensor, True
                            ).to(dtype)
                        else:
                            current_steering = steering_tensor # Already normalized as steering_to_add base

                        if pred_class != 1 or np.abs(distance) < dist_threshold_push:
                            # Wrong class or near boundary: push towards steering direction
                            temp_normalized_activations += current_steering * strength * current_damping
                        elif distance > dist_threshold_pullback:
                            # Correct class, but too far: pull back (opposite of steering direction)
                            temp_normalized_activations -= current_steering * strength * current_damping
                        else:
                            break # Converged
                            
                    # After refinement, update the normalized output base
                    steered_normalized_output = temp_normalized_activations

                elif steer_all_tokens:
                    # Simple application for all tokens (applies to normalized tensor)
                    if 'flux' in args.model_name.lower():
                        alpha = 1.
                    else:
                        alpha = 1
                    if len(steering_to_add.shape) == 3:
                        adjustment_scale = adjustment_scale[:, None]

                    print(adjustment_scale, original_norm.max())

                    

                    steered_normalized_output = alpha * activations_to_modify + (steering_to_add * adjustment_scale).mean(0)

                elif token_indices is not None:
                    # Apply only to selected tokens (applies to normalized tensor)
                    steered_normalized_output = normalized_activations # Start with the original normalized
                    
                    selected_normalized = steered_normalized_output[token_indices]
                    
                    if adjustment_scale.ndim > 1:
                        scaled_steering = steering_to_add[token_indices] * adjustment_scale[token_indices]
                    else:
                        scaled_steering = steering_to_add[token_indices] * adjustment_scale
                        
                    steered_normalized_output[token_indices] = activations_to_modify + scaled_steering
                
            # --- 5. Restore Original Norm (Final Normalization Logic) ---
            
            # Step A: Renormalize the *steered* vector to ensure unit length direction
            vector_direction = steered_normalized_output.float()
            vector_direction = vector_direction / (torch.norm(vector_direction, dim=-1, keepdim=True) + 1e-6)
            print(current_step, layer_idx, calculate_cls_score(vector_direction.cpu().float(), cls_min, model=layer_models[0], cls_type=cls_type, use_distance=False))
            # Step B: Multiply by the original norm to restore magnitude
            vector_restored_norm = vector_direction * original_norm.float()
            
            # Step C: Assemble the final output tuple
            new_output_embeddings = vector_restored_norm.to(dtype)
            

            if double_block:
                if 'flux' in model_type.lower():
                    activations_tuple[output_idx][0] = new_output_embeddings
                else:
                    activations_tuple[output_idx][1] = new_output_embeddings

                new_output = (
                    activations_tuple[0], # Keep QKV output untouched
                    activations_tuple[1], # Use the norm-restored embeddings
                )
            else:
                output[0, :512] = new_output_embeddings
                new_output = output
            
            if 'flux' in model_type.lower():
                layers_idx_tg = 56
            else: 
                layers_idx_tg = 23
            
            if layer_idx == layers_idx_tg:
                current_step += 1
            
            return new_output
        return hook

    # 2. Register Hooks and Processors
    
    # Register hooks on attention layers (attn and attn2 blocks)
    layer_counter = 0
    modules_to_hook = []

    # Get both self-attention ('attn') and cross-attention ('attn2') modules
    for name, module in pipe.transformer.named_modules():
        if name.endswith("attn"):
            modules_to_hook.append(module)
        
            
    for module in modules_to_hook:
        # Register steering hook
        hook_handles.append(module.register_forward_hook(steering_hook(layer_counter)))
        
        # Configure the JointAttnProcessor2_Injection
        if 'flux' not in model_type.lower():
            if hasattr(module, "processor"):
                is_structure_block = layer_counter <= block_structure
                module.processor = JointAttnProcessor2_Injection(
                    do_structure_control=is_structure_block,
                    do_appearance_control=is_structure_block,
                    layer_idx=layer_counter,
                    block=layer_counter,
                    structure_target=['key', 'query'],
                    t_threshold=t_structure,
                )
            
        layer_counter += 1


        if 'flux' not in model_type.lower():
            layer_counter_2 = 0
            modules_to_hook_2 = []

            # Get both self-attention ('attn') and cross-attention ('attn2') modules
            for name, module in pipe.transformer.named_modules():
                if name.endswith("attn2"):
                    modules_to_hook_2.append(module)
                
                    
            for module in modules_to_hook_2:
                # Register steering hook
                
                # Configure the JointAttnProcessor2_Injection
                if hasattr(module, "processor"):
                    is_structure_block = layer_counter <= block_structure
                    module.processor = JointAttnProcessor2_Injection(
                        do_structure_control=is_structure_block,
                        do_appearance_control=is_structure_block,
                        layer_idx=layer_counter,
                        block=layer_counter,
                        structure_target=['key', 'query'],
                        t_threshold=t_structure,
                    )
                
                layer_counter_2 += 1

    def remove_hooks():
        """Function to clean up all registered hooks."""
        for h in hook_handles:
            try:
                h.remove()
            except Exception:
                pass
    
    return step_callback, remove_hooks


# ==============================================================================
# ------------------------------ Execution Block -------------------------------
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply attention steering with injection.")
    
    # --- Model and Generation Args ---
    parser.add_argument('--model_name', type=str, default="stabilityai/stable-diffusion-3.5-medium", help='Stable Diffusion model name or path')
    parser.add_argument('--prompt', type=str, default="a nice blue eyed woman with black hair", help='Prompt for generation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--inference_steps', type=int, default=20, help='Number of inference steps')
    parser.add_argument('--guidance_scale', type=float, default=4.5, help='Guidance scale')
    parser.add_argument('--results_dir', type=str, default='results_block/gothic_2', help='Directory to save results')
    parser.add_argument('--task', type=str, default='add concept', help='')

    # --- Steering Data Args ---
    parser.add_argument('--data_dir', type=str, default='steering_vectors/style/anime', help='Path to data')
    parser.add_argument('--threshold', type=float, default=0.85, help='Threshold for SVM (used in data filenames)')
    parser.add_argument('--n_samples', type=int, default=25, help='Number of samples per class (used in data filenames)')
    
    # --- Steering Control Args ---
    parser.add_argument('--strength', type=float, default=25, help='Steering strength (alpha)')
    parser.add_argument('--block_steering', type=str, default='all', help='Block index or range for steering ("all" or comma-separated list like "0,1,2")')
    parser.add_argument('--t_steering', default='all', help='T threshold for steering')
    parser.add_argument('--orthogonal_projection', action='store_true', help='Use orthogonal projection for steering.')
    parser.add_argument('--iterative_refinement', action='store_true', help='Use iterative refinement with SVM feedback.')
    parser.add_argument('--cls_min', type=float, default=20.0, help='Max penalty score from SVM.')
    
    # --- Token/Quantile Args ---
    parser.add_argument('--best_tokens', action='store_true', help='Use per-token SVM for token selection.')
    parser.add_argument('--best_blocks', action='store_true', help='Use per-block scores and quantiles for masking.')
    parser.add_argument('--quantile_type', type=str, default='no', choices=['no', 'block', 'timestep'], help='Type of quantile masking.')
    parser.add_argument('--quantile_level', type=float, default=0.5, help='Quantile level for masking.')

    # --- Structure Control Args (for JointAttnProcessor2_Injection) ---
    parser.add_argument('--structure', type=float, default=0.5, help='Structure strength (unused in current code, but passed to pipe)')
    parser.add_argument('--block_structure', type=int, default=15, help='Max block index for structure control.')
    parser.add_argument('--t_structure', type=int, default=0, help='T threshold for structure control.')
    
    # --- Data Path Flags (for loading) ---
    parser.add_argument('--separate_normals', action='store_true', help='Separate normals file used.')
    parser.add_argument('--save_svm', action='store_true', help='SVM models file should be loaded.')
    parser.add_argument('--mask_path', type=str, default=None, help='Path to mask file (unused in provided code)')
    parser.add_argument('--photo_path', type=str, default=None, help='Path to real photo for unconditioning/structure')
    parser.add_argument('--prompts_path', type=str, default='data/coco_captions_2017/coco_val2017_subset_250_seed42.txt', help='Path to prompts')
    parser.add_argument('--num_prompts', type=int, default=500, help='number of prompts')

    parser.add_argument('--cls_type', type=str, default='tanh', choices=['steep', 'tanh'], help='Type of SVM penalty function to use (steep or tanh).')

    args = parser.parse_args()

    # Load pipeline
    if 'flux' in args.model_name.lower():
        pipe = FluxPipeline.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16, # bfloat16 is often recommended for Flux
            device_map="balanced",
            use_safetensors=True
        )
    else:
        pipe = StableDiffusion3Pipeline.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,
            device_map="balanced"
        )

    # --- 1. Determine File Paths for Steering Data ---
    p = 'base'
    if args.best_tokens: p += '_best_tokens'

    # Construct file paths based on flags
    data_base_name = f'{p}_{args.threshold}_{args.n_samples}'
    
    # Normal/Steering Vector Path
    normal_suffix = '_normals_separate.pt' if args.separate_normals else '_diff.pt'
    steering_vector_path = os.path.join(args.data_dir, data_base_name + normal_suffix)
    vector = torch.load(steering_vector_path)

    # SVM Model Path
    svm_model_path = os.path.join(args.data_dir, data_base_name + '_svm_models.pt') if args.save_svm else None
    
    # Best Tokens Path
    token_best_path = os.path.join(args.data_dir, data_base_name + '.pt') if args.best_tokens else None
    
    # Scores/Mask Path
    scores_path = os.path.join(args.data_dir, data_base_name + '_scores.pt') if args.best_blocks else None
    
    # --- 2. Load Prompts and Setup Generation ---
    os.makedirs(args.results_dir, exist_ok=True)

    # Load prompts from a file (e.g., animals_prompt.txt)
    coco_captions_path = args.prompts_path
    try:
        with open(coco_captions_path, "r") as f:
            coco_prompts = [line.strip() for line in f if line.strip()]
        coco_prompts = coco_prompts[:args.num_prompts]
    except FileNotFoundError:
        print(f"Warning: Prompt file '{coco_captions_path}' not found. Using default prompt.")
        coco_prompts = [args.prompt]

    # Parse block steering indices
    block_steering: Union[str, List[int]]
    if args.block_steering == '':
        block_steering = []
    else:
        if args.block_steering.lower() != 'all':
            try:
                block_steering = [int(x.strip()) for x in args.block_steering.split(',')]
            except ValueError:
                print("Warning: Invalid format for --block_steering. Using 'all'.")
                block_steering = 'all'
        else:
            block_steering = 'all'

    t_steering: Union[str, List[int]]
    if args.t_steering == '':
        t_steering = []
    else:
        if args.t_steering.lower() != 'all':
            try:
                t_steering = [int(x.strip()) for x in args.t_steering.split(',')]
            except ValueError:
                print("Warning: Invalid format for --block_steering. Using 'all'.")
                t_steering = 'all'
        else:
            t_steering = 'all'

    seeds = [10, 20, 30, 40, 50, 60]

    # # ################
    # seed = 42
    # generator = torch.Generator(device="cpu").manual_seed(seed)

    # p = "Two kids are playing baseball in Wii Sports anime style"
    # res = pipe(p, num_inference_steps=20, guidance_scale=0.0, generator=generator,structure_strength=args.structure, photo=None).images[0]
    # os.makedirs('test', exist_ok=True)
    # from torchvision.utils import make_grid
    # from torchvision.transforms import ToTensor
    # from PIL import Image
    # to_tensor = ToTensor()
    
    # # Determine the best grid layout (closest to square)
    # def save_grid_image(images: List[Image.Image], filename: str):
    #     if not images: return
    #     n = len(images)
    #     nrow = int(n**0.5) # Square root for grid size
    #     tensors = [to_tensor(img) for img in images]
    #     grid = make_grid(tensors, nrow=nrow, padding=2, normalize=False)
    #     # Convert grid tensor back to PIL Image
    #     img = Image.fromarray((grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    #     img.save(os.path.join('test', filename))

    # save_grid_image(
    #     [res], 
    #     f"positive_{0}_{0}_grid.png"
    # )
    # assert False

    # --- 3. Run Generation Loop ---
    for idx, prompt in enumerate(coco_prompts):
        #prompt = "a nice blue eyed woman with black hair"
        # if 'flux' in args.model_name.lower():
        #     prompt = ''

        #prompt = 'dog'
        seed = args.seed
       
        if args.task == 'remove':
            prompt = "A photo of a cool Snoopy" #"A photo of a cool Snoopy" #'A cartoon Snoopy'
            seed = seeds[idx]
            #prompt = 'sketches, pencil_drawing style ' + prompt
        
        #prompt += ' anime style'
        
        # Setup the steering hooks and callback
        step_callback, remove_hooks = apply_attention_steering(
            pipe,
            svm_model_path=svm_model_path,
            scores_path=scores_path,
            token_best_path=token_best_path,
            steering_vectors=vector,
            strength=args.strength,
            block=block_steering,
            t_steering=t_steering,
            t_structure=args.t_structure,
            block_structure=args.block_structure,
            cls_min=args.cls_min,
            orthogonal_projection=args.orthogonal_projection,
            iterative_refinement=args.iterative_refinement,
            quantile_level=args.quantile_level,
            quantile_type=args.quantile_type,
            cls_type=args.cls_type,
            model_type=args.model_name,
            task=args.task
        )
        
        print(f"Processing prompt {idx + 1}/{len(coco_prompts)}: {prompt}")
    
        # Set up generator
        print(seed,args.guidance_scale)
        
        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
        
        # Prepare photo for unconditioning (if provided)
        photo = None
        if args.photo_path is not None:
            img = Image.open(args.photo_path).convert("RGB")
            transform = T.Compose([
                T.ToTensor(), 
                T.Lambda(lambda x: x * 2.0 - 1.0)
            ])
            photo = transform(img).unsqueeze(0).to(pipe.device, dtype=pipe.dtype)
        
       # Run the generation pipeline
        
        image = pipe(
            prompt,
            num_inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            structure_strength=args.structure,
            photo=photo,
            callback=step_callback, # Register the step callback here
            callback_steps=1,
        ).images

        #Clean up hooks
        remove_hooks()
        

        # image = pipe(
        #     prompt,
        #     num_inference_steps=args.inference_steps,
        #     guidance_scale=args.guidance_scale,
        #     generator=generator,
        # ).images
        #assert False

        # --- 4. Save Results ---
        # Construct a descriptive filename based on active arguments
        name_parts = ['base']
        if args.best_tokens: name_parts.append('best_t')
        if args.best_blocks: name_parts.append('best_b')
        if args.save_svm: name_parts.append('cls')
        if args.orthogonal_projection: name_parts.append('ortho')
        if args.iterative_refinement: name_parts.append('iter')
        if args.photo_path: name_parts.append('photo')
        
        name_suffix = "_".join(name_parts)

        # Sanitize prompt for filename
        sanitized_prompt = prompt.replace(" ", "_").replace("/", "").replace(",", "")[:50]
        
        result_filename = (
            f"{idx:02d}_{sanitized_prompt}_s{args.strength}_b{args.block_steering}_"
            f"str{args.structure}_st_{args.t_steering}_cls{args.cls_min}_"
            f"q{args.quantile_type}{args.quantile_level}_{name_suffix}_diff_normed_noisy2.png"
        )
        os.makedirs(os.path.join(args.results_dir, 'steered'), exist_ok=True)
        image[0].save(os.path.join(args.results_dir, 'steered', result_filename))

        if len(image) > 1:
            if args.task == 'remove':
                base_filename = f"{prompt}_{seed}.png"
            else:
                base_filename = f"{prompt}.png"
            os.makedirs(os.path.join(args.results_dir, 'origin'), exist_ok=True)
            image[1].save(os.path.join(args.results_dir, 'origin', base_filename))

        #assert False
        #if 'flux' in args.model_name.lower():
