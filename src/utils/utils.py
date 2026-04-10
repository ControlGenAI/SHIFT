# utils.py

import torch
import numpy as np
import math
from typing import Dict, Any, Optional, List, Tuple, Union
import os
import torch.nn.functional as F
import sklearn.svm._classes


def norm_based_steering_f(current_output: torch.Tensor, norm_type='min') -> torch.Tensor:
    """
    Calculates a scale factor inversely proportional to the norm of each token.
    Tokens with smaller norms get a larger scale factor.

    Args:
        current_output: The activation tensor [L, D].

    Returns:
        A scale factor tensor [L, 1].
    """
    # Norms are [L, 1]
    token_norms = torch.norm(current_output, dim=-1, keepdim=True) + 1e-6
    inverse_norms = 1.0 / token_norms
    
    # Scale based on the minimum inverse norm to keep the scaling factor centered
    min_val = inverse_norms.min()
    max_val = inverse_norms.max()

    if norm_type == 'min':
        scale_factor = (inverse_norms) / (min_val + 1e-6)
    elif norm_type == 'max':
        scale_factor = (inverse_norms) / (max_val + 1e-6)
    elif norm_type == 'min-max':
        scale_factor = (inverse_norms - min_val) / (max_val - min_val + 1e-6)
    elif norm_type == 'no':
        scale_factor = inverse_norms
    else:
        raise ValueError("No such type of normalization")

    return scale_factor



def calculate_cls_score(
    a: torch.Tensor, 
    cls_min: float, 
    model: Optional[Any], 
    cls_type: str = 'steep',  # 'steep' or 'tanh'
    use_distance: bool = False,
    task='add'
):
    """
    Calculates the penalty score based on the SVM decision function distance, 
    using one of two defined penalty functions.

    Args:
        a: The input activation vector (e.g., mean token embedding) [1, D].
        cls_min: The maximum score (penalty) allowed.
        model: The trained SVM model (or list of models).
        cls_type: The type of penalty function to use ('steep' or 'tanh').
        use_distance: If True, use the SVM decision function distance.

    Returns:
        Tuple: (final_score, probability_score, distance)
    """
    #print(cls_min)
    if not use_distance:
        a = (a / a.norm(dim=-1))#.mean(1)
        # a = model['scaler'].transform(a)
        # a = model['pca'].transform(a)
        score_cls = model.predict_proba(a)  if model is not None else np.array([[0.5, 0.5]])
        if task == 'add concept':
            scoreeee =  min(cls_min, (1 / ((1 - score_cls[0][0]) + 1e-8) - 1)) # 21 * score_cls[0][1]**2
            #print(score_cls, scoreeee)
        else:
            #assert False
            scoreeee = (1 / ((1 - score_cls[0][1]) + 1e-8) - 1).clip(0, cls_min) #  21 * score_cls[0][1]**2  
            #print('dd', scoreeee)
        return scoreeee, score_cls[0][0], None

    model_instance = model[0] if isinstance(model, (list, tuple)) else model
    
    distance = model_instance.decision_function(a.cpu())[0]

    if cls_type == 'steep':
        scale_correct = 1.0
        decay_rate = 2.0
        wrong_class_offset = 1.0 
        scale_wrong = 5.0 

        if distance >= 0:
            # Correct class: Decaying exponential (Penalty approaches 0 as distance increases)
            scoreeee = scale_correct * math.exp(-decay_rate * distance)
        else:
            # Wrong class: Steep Linear penalty (Penalty increases rapidly as distance decreases)
            scoreeee = wrong_class_offset - scale_wrong * distance
            
        prob_score = 0.0 # Mock or unused in this specific function

    elif cls_type == 'tanh':
        # --- Tanh Penalty (Good for penalizing proximity to the boundary) ---
        scale = 1.0
        offset = 0.0
        
        # Score is high when distance is low (near the boundary)
        scoreeee = max(0.0, (1.0 - math.tanh((distance + offset) / scale)))
        
        prob_score = model_instance.predict_proba(a.cpu())[0][0] 

    else:
        raise ValueError(f"Unknown cls_type: {cls_type}. Must be 'steep' or 'tanh'.")

    final_score = min(cls_min, scoreeee)
    return final_score, prob_score, distance


def apply_txt_steering(pooled_prompt_embeds, prompt_embeds, pooled_style, seqs_style, normed=True, strength=0.0, task='add'):
    #normed = True
    

    if strength != 0.0:
        if task == 'add concept':
            scale = 1
        else:
            scale = F.cosine_similarity(pooled_prompt_embeds.clone() / pooled_prompt_embeds.norm(dim=-1, keepdim=True), pooled_style.clone() / pooled_style.norm(dim=-1, keepdim=True), dim=-1)[0]
            if task != 'add concept':
                scale = -scale
            assert False
            
            scale = scale.clip(0,1)
    else:
        scale = 0.0

    #print(scale, pooled_style.shape, seqs_style.shape)
    
    if not normed:
        new_pooled_embeds = pooled_prompt_embeds + pooled_style * scale
        new_prompt_embeds = prompt_embeds + seqs_style
    else:
        
        init_pooled_norm = pooled_prompt_embeds.clone().norm(dim=-1, keepdim=True)
        init_prompt_norm = prompt_embeds.clone().norm(dim=-1, keepdim=True)

        new_pooled_embeds = (pooled_prompt_embeds +  pooled_style)
        new_pooled_embeds = new_pooled_embeds / new_pooled_embeds.norm(dim=-1, keepdim=True) * init_pooled_norm

        new_prompt_embeds = (prompt_embeds +  seqs_style)
        new_prompt_embeds = new_prompt_embeds / new_prompt_embeds.norm(dim=-1, keepdim=True) * init_prompt_norm


    return new_pooled_embeds, new_prompt_embeds, scale


def steering_txt_data(vector_txt, strenght, prompt_embeds, mean=True, ssim=False, pooled=True, normed=True, seq=False, task='add'):
    seqs_style = vector_txt['sequence'].to(prompt_embeds.dtype).to(prompt_embeds.device)
    if len(seqs_style.shape) == 2:
        seqs_style = seqs_style.unsqueeze(0)

    seqs_style = seqs_style.mean(0, keepdim=True) 
    pooled_style = vector_txt['pooled'].to(prompt_embeds.dtype).to(prompt_embeds.device)
    if len(pooled_style.shape) == 1:
        pooled_style = pooled_style.unsqueeze(0)

    pooled_style = pooled_style.mean(0, keepdim=True) 
    
    mean = False
    
    print(seqs_style.shape, pooled_style.shape, mean)
    if mean:
        seqs_style = seqs_style.mean(1, keepdim=True) 
    if ssim:
        sim_add = F.cosine_similarity(prompt_embeds.clone() / prompt_embeds.norm(dim=-1, keepdim=True), seqs_style.clone() / seqs_style.norm(dim=-1, keepdim=True), dim=-1)[0]
        k_ratio = 0.1
        k_val = int(sim_add.shape[0] * k_ratio)
        
        if k_val > 0:
            # Find the threshold value for the top K
            threshold = torch.topk(sim_add.flatten(), k_val).values[-1]
            
            sim_mask = (sim_add >= threshold).float()
        else:
            # Fallback for very short sequences
            sim_mask = (sim_add > 0.1).float()

        seqs_style = seqs_style * sim_mask.unsqueeze(1).to(seqs_style.dtype)

    if normed:
        if pooled:
            pooled_style = pooled_style / pooled_style.norm(dim=-1, keepdim=True)
        seqs_style = seqs_style / seqs_style.norm(dim=-1, keepdim=True)
    
    seqs_style = seqs_style * strenght
    
    if pooled:
        pooled_style = pooled_style * strenght
    else:
        pooled_style = pooled_style * 0.
    seq = True
    if not seq:
        seqs_style = torch.zeros_like(seqs_style).to(pooled_style.device).to(pooled_style.dtype)

    if task != 'add concept':
        assert False
        pooled_style = -pooled_style
        seqs_style = -seqs_style

    return pooled_style, seqs_style

def load_steering_data(
    svm_model_path: Optional[str], 
    scores_path: Optional[str], 
    token_best_path: Optional[str], 
    quantile_level: float
) -> Tuple[Optional[Dict], Optional[torch.Tensor], Optional[torch.Tensor], Optional[np.ndarray]]:
    """Loads SVM models, best tokens, and scores using original logic."""
    
    models = None
    if svm_model_path and os.path.exists(svm_model_path):
        with torch.serialization.safe_globals([sklearn.svm._classes.SVC]): 
            models = torch.load(svm_model_path, weights_only=False) 

    tokens_best = None
    if token_best_path and os.path.exists(token_best_path):
        tokens_best = torch.load(token_best_path, weights_only=False)
     
    scores_all = None
    if scores_path and os.path.exists(scores_path):
        # Handle both tensor and numpy loads safely
        data = torch.load(scores_path, weights_only=False)
        scores_all = data if torch.is_tensor(data) else torch.from_numpy(data)

    quantiles = None
    if scores_all is not None and scores_path:
        scores_np = scores_all.numpy()
        if 'block' in scores_path:
            quantiles = np.quantile(scores_np, q=quantile_level, axis=2) 
        elif 'timestep' in scores_path:
            quantiles = np.quantile(scores_np, q=quantile_level, axis=(1, 2)) 
        else:
            quantiles = np.quantile(scores_np, q=quantile_level)
    
    return models, tokens_best, scores_all, quantiles