import torch
import numpy as np
import os
import torch.nn.functional as F
import argparse
from PIL import Image
import torchvision.transforms as T
from sd_3 import StableDiffusion3Pipeline
from ctrlx.sd_3_injection import JointAttnProcessor2_Injection
import sklearn.svm._classes
import math
from tqdm import tqdm




def orthogonal_projection_steering(current_output, steering_tensor, normalize=True):

    dot_product_numerator = torch.sum(steering_tensor * current_output, dim=1, keepdim=True)
    projection_vector = dot_product_numerator * current_output 

    # 4. Calculate the orthogonal component
    orthogonal_steering = steering_tensor - projection_vector

    if normalize:
        orthogonal_steering = orthogonal_steering / torch.norm(orthogonal_steering, dim=-1, keepdim=True)

    return orthogonal_steering


def norm_based_steering_f(current_output):
    token_norms = torch.norm(current_output, dim=-1, keepdim=True) + 1e-6
    inverse_norms = 1.0 / token_norms
    min_val = inverse_norms.min()
    max_val = inverse_norms.max()
    scale_factor = (inverse_norms) / (min_val + 1e-6)
    return scale_factor


def cls_score_revised_steep(a, cls_min, layer_idx, current_step, model, use_distance=True):
    '''The revised two-part penalty function with a steeper wrong-class penalty.'''
    if not use_distance or model is None:
        # Simplified prediction path for non-distance calculation (not used in plot)
        if model is None: return 0, 0, None 
        score_cls = model.predict_proba(a)
        scoreeee = min(cls_min, (1 / ((1 - score_cls[0][0]) + 1e-8) - 1))
        return scoreeee, score_cls[0][0], None

    # Tunable Parameters
    scale_correct = 1.0
    decay_rate = 2.0
    
    # --- MODIFIED PARAMETERS FOR STEEPER PENALTY ---
    # wrong_class_offset remains 1.0 for continuity at D=0
    wrong_class_offset = 1.0 
    # Increased scale_wrong from 1.5 to 5.0 for a much steeper linear penalty
    scale_wrong = 5.0 
    # ------------------------------------------------
    
    distance = model.decision_function(a)
    
    if distance >= 0:
        # Correct class: Decaying exponential (Penalty approaches 0 as distance increases)
        scoreeee = scale_correct * math.exp(-decay_rate * distance)
    else:
        # Wrong class: Steep Linear penalty (Penalty increases rapidly as distance decreases)
        scoreeee = wrong_class_offset - scale_wrong * distance

    final_score = min(cls_min, scoreeee)
    prob_score = 0 # Mocking the prob score
    return final_score, prob_score, distance




def cls_score(a, cls_min, layer_idx, current_step, model=None, use_distance=False):
    '''
    some other ideas
    power = 1
    prob_term = score_cls[0][0]
    fused = (1 - math.tanh(distance / scale)) * prob_term**power

    if layer_idx <= 12:
        decay_factor = 1
    else:
        decay_factor = 1 #0.5 * (1 + np.cos(np.pi * layer_idx / 23))

    '''
    score_cls = model.predict_proba(a) if model is not None else np.array([[0.5, 0.5]])

    if not use_distance:
        #assert False
        scoreeee = (min(cls_min, (1 / ((1-score_cls[0][0]) + 1e-8) - 1))) #score_cls[0][0] #
        return scoreeee, score_cls, None
    
    distance = model.decision_function(a.cpu())[0]
    scale = 1
    offset = 0
    scoreeee =  min(cls_min, max(0, (1 - math.tanh((distance + offset) / scale))))
    return scoreeee, score_cls[0][0], distance



def apply_attention_steering(
    pipe,
    svm_model_path=None,
    scores_path=None,
    mask_path=None,
    token_best_path=None,
    steering_vectors=None,
    strength=1.0,
    block='all',
    t_structure=0,
    t_steering=0,
    block_structure=30,
    activations='attn_enc',
    task='add concept',
    cls_min=20, 
    block_threshold=1,
    iterative_refinement=False,
    orthogonal_projection=False,
    norm_based_steering=False,
    quantile_type='block',
    quantile_level=0.5,
):
    """
    Apply attention steering vectors during generation with adjustable strength
    Args:
        pipe: StableDiffusionPipeline
        svm_model_path: Path to SVM models
        scores_path: Path to scores file
        mask_path: Path to mask file
        token_best_path: Path to best token indices
        steering_vectors: Dict of steering vectors (from get_attention_steering_vector_multistep)
        strength: Float (positive/negative) to control steering intensity
    """
    hook_handles = []
    current_step = 0

    print('==================', t_steering, '===================') 
    with torch.serialization.safe_globals([sklearn.svm._classes.SVC]):
        models = torch.load(svm_model_path, weights_only=False) if svm_model_path and os.path.exists(svm_model_path) else None

    tokens_best = torch.load(token_best_path, weights_only=False) if token_best_path and os.path.exists(token_best_path) else None
     
    if quantile_type == 'no':
        scores_path = None

    scores_all = torch.from_numpy(torch.load(scores_path, weights_only=False)) if scores_path and os.path.exists(scores_path) else None
    
    if scores_all is not None:
        if quantile_type == 'block':
            quantiles = np.quantile(scores_all.numpy(), q=quantile_level, axis=0)
        elif quantile_type == 'timestep':
            quantiles = np.quantile(scores_all.numpy(), q=quantile_level, axis=1)
        else:
            quantiles = np.quantile(scores_all.numpy(), q=quantile_level)
        print(scores_all, quantiles.shape, quantiles, quantile_level, scores_all.shape)
    
   
    def step_callback(step_idx, timestep, latents,  **kwargs):
        nonlocal current_step
        current_step = step_idx

    def steering_hook(layer_idx):
        def hook(module, input, output):
            nonlocal current_step
            device = output[0].device
            dtype = output[0].dtype
        
            # Only apply if we have a steering vector for this step and layer
            if current_step in steering_vectors and f"layer_{layer_idx}" in steering_vectors[current_step]:
                #assert False
                vec = steering_vectors[current_step][f"layer_{layer_idx}"]
                model = models[current_step][f"layer_{layer_idx}"] if models is not None else None
                
                try:
                    len_models = len(model)
                    print(len_models)
                except:
                    model = [model]
                    len_models = 1

                jjj = 1 if activations == 'attn_enc' else 0
                
                if tokens_best is None or len(tokens_best[current_step][layer_idx]) != 0:
                    if isinstance(vec, list) or (hasattr(vec, 'shape') and len(vec.shape) > 1):
                        steering_tensor = vec.to(device=device, dtype=dtype)
                    else:
                        steering_tensor = vec.to(device=device, dtype=dtype)


                    if activations == 'attn_enc' or activations == 'attn_im':
                        norm = torch.norm(output[jjj].float(), dim=-1, keepdim=True)
                        
                        if task == 'add concept':
                                if tokens_best is None or len(tokens_best[current_step][layer_idx]) != 0:
                                    v_norm = torch.norm(steering_tensor, dim=-1, keepdim=True)
                                    steering_tensor = steering_tensor / (v_norm + 1e-6)
                                
                                if block == 'all' or (hasattr(block, '__contains__') and layer_idx in block):
                                    print('hello')
                                    if tokens_best is None or len(tokens_best[current_step][layer_idx]) != 0:
                                        if model is not None:

                                            print(model)
                                            
                                            a = output[jjj][1][tokens_best[current_step][layer_idx]].mean(0)[None].cpu().clone() if tokens_best is not None else output[jjj][1].mean(0)[None].cpu().clone()
                                            a = a / a.norm(dim=-1)
                                            scoreeee_all = []
                                            distance_all = []
                                            score_cls_all = []
                                            
                                            for i in range(len(model)):
                                                scoreeee, score_cls, distance = cls_score(a, cls_min, layer_idx, current_step, model=model[i], use_distance=True)
                                                scoreeee_all.append(scoreeee)
                                                distance_all.append(distance)
                                                score_cls_all.append(score_cls) 
                                            
                                            scoreeee = torch.tensor(scoreeee_all).unsqueeze(1).to(device) #np.mean(scoreeee_all)
                                            print(scoreeee_all, scoreeee, score_cls, cls_min, distance_all) 
                                        else:
                                            scoreeee = 1
                                        
                                        if scores_all is not None:
                                            quantiles
                                            if quantile_type == 'block':
                                               score = (scores_all[current_step][layer_idx] >= quantiles[layer_idx]) 
                                            elif quantile_type == 'timestep':
                                                score = (scores_all[current_step][layer_idx] >= quantiles[current_step]) 
                                            else:
                                                score = (scores_all[current_step][layer_idx] >= quantiles) 
                                                                                       
                                        else:
                                            score = 1

                                        
                                        if tokens_best is not None:
                                            if len(steering_tensor.shape) != 1:
                                                assert False
                                                output[jjj][1][tokens_best[current_step][layer_idx]] = output[jjj][1][tokens_best[current_step][layer_idx]] + steering_tensor[tokens_best[current_step][layer_idx]] * strength * scoreeee * score
                                            else:
                                                #print(tokens_best[current_step][layer_idx])
                                                # output[jjj][1] = output[jjj][1] / norm[1]
                                                # print('======================')
                                                # print( 'norm', norm[1])
                                                # print('======================')
                                                #tokens_best[current_step][layer_idx] = tokens_best[current_step][layer_idx].astype(steering_tensor.dtype)
                                                norm_based_scaling = 1
                                                norm_based_steering = True
                                                # orthogonal_projection = True
                                                if norm_based_steering:
                                                    norm_based_scaling = norm_based_steering_f(output[jjj][1].clone())

                                                if orthogonal_projection:
                                                    current_output = output[jjj][1].clone() / norm[1]
                                                    orthogonal_steering = orthogonal_projection_steering(current_output, steering_tensor, True )
                                                    output[jjj][1][tokens_best[current_step][layer_idx]] = (output[jjj][1][tokens_best[current_step][layer_idx]] + orthogonal_steering[tokens_best[current_step][layer_idx]] * strength * scoreeee * score).to(dtype)

                                                else:  
                                                    assert False  
                                                    # if current_step <= 5 and layer_idx <= 5:
                                                    #     torch.save({'activations': output[jjj][1], 'steering_tensor':steering_tensor, 'tokens': tokens_best[current_step][layer_idx]}, f'test_sim/{current_step}_{layer_idx}.pt')
                                                    # * norm_based_scaling[tokens_best[current_step][layer_idx]]
                                                    output[jjj][1][tokens_best[current_step][layer_idx]] = (output[jjj][1][tokens_best[current_step][layer_idx]] + steering_tensor * strength * scoreeee * score ).to(dtype)
                                        else:

                                            if iterative_refinement:
                                                max_iter = 10
                                                dist_threshold = 3.0  # Tunable: distance beyond which we pull back
                                                dist_threshold_1 = 2.0
                                                damping_factor = 0.3  # Tunable: adjustment strength
                                                damping_factor_remove = 0.3
                                                scale = 1.0  # For tanh, from your setup
                                                adjustment = strength * scoreeee * score
                                                
                                                if orthogonal_projection:
                                                        current_output = output[jjj][1].clone() / norm[1]
                                                        
                                                        orthogonal_steering = orthogonal_projection_steering(current_output, steering_tensor, True )
                                                        output[jjj][1] = output[jjj][1] + orthogonal_steering * adjustment

                                                else:
                                                    output[jjj][1] = output[jjj][1]  + steering_tensor * adjustment

                                                steered_output = output[jjj][1].clone()

                                                for iter in tqdm(range(max_iter)):
                                                
                                                    
                                                    a_iter = output[jjj][1][tokens_best[current_step][layer_idx]].mean(0)[None].cpu().clone() if tokens_best is not None else output[jjj][1].mean(0)[None].cpu().clone()
                                                    a_iter = a_iter / a_iter.norm(dim=-1)

                                                    
                                                    # Run SVM classifier
                                                    if model is not None:
                                                        pred_class = model[0].predict(a_iter)[0]  # Class 1 = target (e.g., hat)
                                                        distance = model[0].decision_function(a_iter)[0]  # Raw decision function
                                                        print('==========')
                                                        print(f"Iter {iter}: Class {pred_class}, Distance {distance}")

                                                        if pred_class != 1 or np.abs(distance) < dist_threshold_1:  # Wrong class: push towards steering
                                                           
                                                            if orthogonal_projection:
                                                                print(strength * damping_factor,  strength * (damping_factor * max(0, 1 - math.tanh(distance / scale))), 'add')
                                                                orthogonal_steering = orthogonal_projection_steering(a_iter.to(steering_tensor.device), steering_tensor, True)
                                                                adjustment = orthogonal_steering * strength * damping_factor #(damping_factor * max(0, 1 - math.tanh(distance / scale)))
                                                            else:
                                                                adjustment = steering_tensor * strength * damping_factor #(damping_factor_remove * max(0, 1 - math.tanh(distance / scale)))
                                                            
                                                            steered_output += adjustment
                                                            damping_factor *= 0.5
                                                            
                                                        
                                                        elif distance > dist_threshold:  # Correct class, too far: pull back
                                                            if orthogonal_projection:
                                                                print(strength * damping_factor_remove, strength * (damping_factor_remove * max(0, 1 - math.tanh(distance / scale))), 'remove')
                                                            
                                                                orthogonal_steering = orthogonal_projection_steering(a_iter.to(steering_tensor.device), steering_tensor, True)
                                                                adjustment = steering_tensor * strength * damping_factor_remove #(damping_factor_remove * max(0, 1 - math.tanh(distance / scale)))
                                                            
                                                            else:
                                                                adjustment = steering_tensor * strength * damping_factor_remove #(damping_factor_remove * max(0, 1 - math.tanh(distance / scale)))
                                                            
                                                            
                                                            steered_output -= adjustment
                                                            damping_factor_remove *= 0.5
                                                        else:
                                                            break  # Converged: correct class, near plane
                                                    

                                        # Assign refined output
                                                    output[jjj][1] = steered_output
                                            else:
                                                if orthogonal_projection:
                                                    print('here')
                                                    current_output = output[jjj][1].clone() / norm[1]
                                                    
                                                    orthogonal_steering = orthogonal_projection_steering(current_output, steering_tensor, True)
                                                    output[jjj][1] = output[jjj][1] + orthogonal_steering * strength * scoreeee * score

                                                else:
                                                    if len_models == 1:
                                                        output[jjj][1] = output[jjj][1]  + steering_tensor * strength * scoreeee * score
                                                    else:

                                                        output[jjj][1][:] = output[jjj][1][:]  + (steering_tensor[:5] * strength * scoreeee[:5] * score).mean(0)
                                                        print((steering_tensor[:3] * strength * scoreeee[:3] * score).shape)
                                                

                                                        
                                           
                        else:
                            sim = torch.tensordot(output[jjj][1:], steering_tensor[None], dims=([2], [2])).view(output[jjj][1:].size()[0], output[jjj][1:].size()[1], 1)
                            sim = torch.where(sim>0, sim, 0)
                            output[jjj][1] =  output[jjj][1] - (strength*sim)*steering_tensor
                        vector = output[jjj].float()
                        
                        vector = vector / torch.norm(vector, dim=-1, keepdim=True)
                        
                        vector = vector * norm
                        if jjj == 1:
                            new_output = (
                                output[0],
                                vector.to(dtype),
                            )
                        else:
                            new_output = (
                                vector.to(dtype),
                                output[1]
                            )
                else:
                    new_output = output
                if layer_idx == 23:
                    current_step += 1
                    print(current_step)
                return new_output
            return output
        return hook

    # Register hooks on attention layers
    idx_layes = 0
    for idx, (name, module) in enumerate(pipe.transformer.named_modules()):
        if name.endswith("attn"):
            hook_handles.append(module.register_forward_hook(steering_hook(idx_layes)))
            if hasattr(module, "processor"):
                if idx_layes <= block_structure:
                    module.processor = JointAttnProcessor2_Injection(
                        do_structure_control=True,
                        do_appearance_control=True,
                        layer_idx=idx_layes,
                        block=idx_layes,
                        structure_target=['key', 'query'],
                        t_threshold=t_structure,  # Fixed: was t_th
                    )
                else:
                    module.processor = JointAttnProcessor2_Injection(
                        do_structure_control=False,
                        do_appearance_control=False,
                        layer_idx=idx_layes,
                        block=idx_layes,
                        structure_target=['key', 'query'],
                        t_threshold=t_structure,  # Fixed: was t_th
                    )
            idx_layes += 1
    
    for idx, (name, module) in enumerate(pipe.transformer.named_modules()):
        if name.endswith("attn2"):
            #hook_handles.append(module.register_forward_hook(steering_hook(idx_layes)))
            if hasattr(module, "processor"):
                if idx_layes <= block_structure:
                    module.processor = JointAttnProcessor2_Injection(
                        do_structure_control=True,
                        do_appearance_control=True,
                        layer_idx=idx_layes,
                        block=idx_layes,
                        structure_target=['key', 'query'],
                        t_threshold=t_structure,  # Fixed: was t_th
                    )
                else:
                    module.processor = JointAttnProcessor2_Injection(
                        do_structure_control=False,
                        do_appearance_control=False,
                        layer_idx=idx_layes,
                        block=idx_layes,
                        structure_target=['key', 'query'],
                        t_threshold=t_structure,  # Fixed: was t_th
                    )
            idx_layes += 1

    

    def remove_hooks():
        for h in hook_handles:
            try:
                h.remove()
            except Exception:
                pass
    
    return step_callback, remove_hooks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply attention steering with injection.")
    parser.add_argument('--model_name', type=str, default="stabilityai/stable-diffusion-3.5-medium", help='Stable Diffusion model name or path')
    parser.add_argument('--data_dir', type=str, default='steering_vectors/style/anime', help='Path to data')
    parser.add_argument('--prompt', type=str, default="a nice blue eyed woman with black hair", help='Prompt for generation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--inference_steps', type=int, default=20, help='Number of inference steps')
    parser.add_argument('--guidance_scale', type=float, default=4.5, help='Guidance scale')
    parser.add_argument('--results_dir', type=str, default='results_block/gothic_2', help='Directory to save results')
    parser.add_argument('--block_threshold', type=float, default=0.85, help='Steering strength')
    parser.add_argument('--strength', type=float, default=25, help='Steering strength')
    parser.add_argument('--structure', type=float, default=0.5, help='Structure strength')
    parser.add_argument('--block_structure', type=int, default=15, help='Block structure value')
    parser.add_argument('--t_structure', type=int, default=0, help='T threshold for structure')
    parser.add_argument('--t_steering', type=int, default=0, help='T threshold for steering')
    parser.add_argument('--block_steering', type=str, default='all', help='Block ("all" or range)')
    parser.add_argument('--best_tokens', action='store_true', help='Whether to compute best tokens using per-token SVMs')
    parser.add_argument('--best_blocks', action='store_true', help='Whether to compute best tokens using per-token SVMs')
    parser.add_argument('--separate_normals', action='store_true', help='Whether to save normals and scores for best tokens separately')
    parser.add_argument('--save_svm', action='store_true', help='Whether to save SVMs and normals for best tokens')
    parser.add_argument('--threshold', type=float, default=0.85, help='Guidance scale (unused)')
    parser.add_argument('--n_samples', type=int, default=25, help='Number of samples per class')
    parser.add_argument('--mask_path', type=str, default=None, help='Path to mask file')
    parser.add_argument('--photo_path', type=str, default=None, help='Path to real photo to generate')
    parser.add_argument('--orthogonal_projection', action='store_true', help='Whether to save normals and scores for best tokens separately')
    parser.add_argument('--iterative_refinement', action='store_true', help='Whether to save normals and scores for best tokens separately')
    parser.add_argument('--quantile_type', help='Whether to save normals and scores for best tokens separately')
    parser.add_argument('--quantile_level', type=float, default=0.5, help='Structure strength')
    args = parser.parse_args()

    # Load pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="balanced"
    )

    p = 'base'

    if args.best_tokens:
        p = p + '_best_tokens'

    if args.separate_normals:
        steering_vector_path = os.path.join(args.data_dir, f'{p}_{args.threshold}_{args.n_samples}_normals_separate.pt')
    else:
        steering_vector_path = os.path.join(args.data_dir, f'{p}_{args.threshold}_{args.n_samples}_normals.pt')
    #steering_vector_path = '/home/jovyan/shares/SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs/steering_vectors/plane/style/normals/normals_plane_50_norm_anime.pt' #'/home/jovyan/shares/SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs/steering_vectors/plane/style/normals/normals_plane_25_norm_digital_art_best_tokens_85.pt'
    vector = torch.load(steering_vector_path)
    #vector_1 = torch.load('/home/jovyan/shares/SR006.nfs2/konovalova/workspace/attention-map-diffusers/steering_vecs/steering_vector/add/hat_peop_ansamble/base_0.85_20_normals.pt')
    
    #vector = {0: vector, 1: vector_1}
    
    
    os.makedirs(args.results_dir, exist_ok=True)

    svm_model_path = None
    token_best_path = None
    scores_path = None

    if args.save_svm:
        svm_model_path = os.path.join(args.data_dir, f'{p}_{args.threshold}_{args.n_samples}_svm_models.pt')
    
    if args.best_tokens:
        
        token_best_path = os.path.join(args.data_dir, f'{p}_{args.threshold}_{args.n_samples}.pt')
        print('beeeeest tokens' , token_best_path)
    if args.best_blocks:
        scores_path = os.path.join(args.data_dir, f'{p}_{args.threshold}_{args.n_samples}_scores.pt')



    # INSERT_YOUR_CODE

    # Load 50 prompts from coco_captions.txt
    #coco_captions_path ="./coco_captions.txt"
    #coco_captions_path = 'add_concepts_prompts.txt'
    coco_captions_path = 'animals_prompt.txt'
    #coco_captions_path = 'animals_diff_prompts.txt'
    #coco_captions_path = 'simple_prompts_add.txt'
    #coco_captions_path = 'simple_prompts_people_age.txt'
    #coco_captions_path = 'simple_prompts_woman.txt'
    with open(coco_captions_path, "r") as f:
        coco_prompts = [line.strip() for line in f if line.strip()]
    coco_prompts = coco_prompts[:35]

    if args.block_steering != 'all':
        block_steering = [int(x) for x in args.block_steering.split(',')]
    else:
        block_steering = args.block_steering

    for idx, prompt in enumerate(coco_prompts):
        step_callback, remove_hooks = apply_attention_steering(
            pipe,
            svm_model_path=svm_model_path,
            scores_path=scores_path,
            mask_path=args.mask_path,
            token_best_path=token_best_path,
            steering_vectors=vector,
            strength=args.strength,
            block=block_steering,
            t_structure=args.t_structure,
            t_steering=args.t_steering,
            block_structure=args.block_structure,
            block_threshold=args.block_threshold,
            orthogonal_projection=args.orthogonal_projection,
            iterative_refinement=args.iterative_refinement,
            quantile_level=args.quantile_level,
            quantile_type=args.quantile_type,

        )
        #prompt = ''

        print('process ', prompt)
        # Set up generator and generate
        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(args.seed)
        
        # Register callback
        #pipe.scheduler.set_timesteps(args.inference_steps)

        # INSERT_YOUR_CODE
        photo = None
        if args.photo_path is not None:

            img = Image.open(args.photo_path).convert("RGB")
            transform = T.Compose([
                T.ToTensor(),  # Converts to [0,1]
                T.Lambda(lambda x: x * 2.0 - 1.0)  # Scale to [-1, 1]
            ])
            photo = transform(img).unsqueeze(0)  # Add batch dimension
        
        image = pipe(
            prompt,
            num_inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            structure_strength=args.structure,
            photo=photo,
        ).images

        # Clean up hooks
        remove_hooks()

        name = '_base'
        if args.best_tokens:
            name = name + '_best_tokens'
        if args.separate_normals:
            name = name + '_separate_normals'
        if args.best_blocks:
            name = name + '_best_blocks'
        if args.save_svm:
            name = name + '_cls'
        if args.photo_path is not None:
            name = name + '_photo'

        # Save results
        ffff = 0
        result_filename = f"{prompt}_strength_{args.strength}_block_{args.block_steering}_block_{args.block_structure}_structure_{args.block_structure}_{args.structure}_{name}_{args.t_structure}.png"
        image[0].save(os.path.join(args.results_dir, result_filename))
        
        # if len(image) > 1:
        #     base_filename = f"{prompt}_{ffff}.png"
        #     #base_filename = f"{prompt}_base_generation_strength_{args.strength}_block_{args.block_steering}_structure_{args.block_structure}_structure_{args.block_structure}_{args.structure}_{name}_{ffff}.png"
        #     image[1].save(os.path.join('steering_metrics/animal_new_back/animal', base_filename))

        # if len(image) > 2:
        #     base_filename = f"{prompt}_photo_generation_strength_{args.strength}_block_{args.block_steering}_structure_{args.block_structure}_{args.structure}_{name}.png"
        #     image[2].save(os.path.join(args.results_dir, base_filename))
        
        print(f"Results saved to {args.results_dir}")
        #assert False
        