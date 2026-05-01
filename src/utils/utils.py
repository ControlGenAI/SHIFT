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
        print(a.shape)
        a = (a / a.norm(dim=-1))#.mean(1)
        # a = model['scaler'].transform(a)
        # a = model['pca'].transform(a)
        score_cls = model.predict_proba(a)  if model is not None else np.array([[0.5, 0.5]])
        if task == 'add concept':
            assert False
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


def _to_pooled_device_dtype(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return t.to(device=ref.device, dtype=ref.dtype)


def _get_pooled_steering_payload_tensor(
    vector_txt: Dict[str, Any], key: str, alts: Tuple[str, ...], ref: torch.Tensor
) -> torch.Tensor:
    for k in (key,) + alts:
        if k in vector_txt and torch.is_tensor(vector_txt[k]):
            return _to_pooled_device_dtype(vector_txt[k], ref)
    raise KeyError(f"pooled steering payload missing {key!r} (tried {alts})")


def pooled_steering_mode(vector_txt: Any) -> str:
    """
    Advanced pooled steering only when explicitly requested on the vector dict:
      pooled_steering_mode in ('subspace_mean', 'monge').
    Any other value (including missing / 'raw' / 'legacy') → 'raw' = обычный стиринг
    через steering_txt_data + apply_txt_steering.
    """
    if not isinstance(vector_txt, dict):
        return "raw"
    mode = vector_txt.get("pooled_steering_mode", "raw")
    if mode in ("subspace_mean", "monge"):
        return str(mode)
    return "raw"


def use_pooled_advanced_txt_steering(txt_steering: Dict[str, Any], vector: Any) -> bool:
    """
    True только для явного advanced-режима. Иначе всегда legacy (pooled+sequence в vector).
    Принудительно legacy: txt_steering['legacy_txt_steering'] == True.
    """
    if txt_steering.get("legacy_txt_steering", False):
        return False
    if not isinstance(vector, dict):
        return False
    return vector.get("pooled_steering_mode") in ("subspace_mean", "monge")


def compute_pooled_steering_delta_advanced(
    pooled_b_c: torch.Tensor,
    vector_txt: Dict[str, Any],
    strength: float,
    task: str,
) -> torch.Tensor:
    """
    Δ pooled (B, C) for subspace mean-diff or Monge transport in a low-d subspace.

    vector_txt keys (tensors, same C as pooled; d = subspace dim):
      - pooled_steering_mode: 'subspace_mean' | 'monge'
      - mu_neg: (C,)  anchor in full space (also tries mu_neg_anchor)
      - V_d: (C, d)  orthonormal columns (tries V)

    subspace_mean additionally:
      - delta_z: (d,) mean-diff direction in z-space (tries mean_diff_delta_z_global / mean_diff_delta_z_train)

    monge additionally:
      - mu_neg_z, mu_pos_z: (d,)
      - M: (d, d)
    """
    mode = pooled_steering_mode(vector_txt)
    if mode == "raw":
        raise ValueError("compute_pooled_steering_delta_advanced: mode is raw")

    ref = pooled_b_c
    mu_neg = _get_pooled_steering_payload_tensor(vector_txt, "mu_neg", ("mu_neg_anchor",), ref)
    V = _get_pooled_steering_payload_tensor(vector_txt, "V_d", ("V",), ref)

    if mu_neg.dim() != 1:
        raise ValueError(f"mu_neg must be (C,), got {tuple(mu_neg.shape)}")
    if V.dim() != 2 or V.shape[0] != mu_neg.shape[0]:
        raise ValueError(f"V_d must be (C, d), C={mu_neg.shape[0]}, got {tuple(V.shape)}")

    if mode == "subspace_mean":
        delta_z = None
        for k in ("delta_z", "mean_diff_delta_z_global", "mean_diff_delta_z_train"):
            if k in vector_txt and torch.is_tensor(vector_txt[k]):
                delta_z = _to_pooled_device_dtype(vector_txt[k], ref)
                break
        if delta_z is None:
            raise KeyError("subspace_mean: need delta_z or mean_diff_delta_z_* (d,) tensor")
        if delta_z.dim() != 1 or delta_z.shape[0] != V.shape[1]:
            raise ValueError(f"delta_z must be (d,) with d={V.shape[1]}, got {tuple(delta_z.shape)}")
        if task != "add concept":
            delta_z = -delta_z
        delta_c = V @ (delta_z * float(strength))
        return delta_c.unsqueeze(0).expand_as(pooled_b_c)

    # monge
    mu_neg_z = _get_pooled_steering_payload_tensor(vector_txt, "mu_neg_z", (), ref)
    mu_pos_z = _get_pooled_steering_payload_tensor(vector_txt, "mu_pos_z", (), ref)
    M = _get_pooled_steering_payload_tensor(vector_txt, "M", (), ref)
    if mu_neg_z.dim() != 1 or mu_pos_z.dim() != 1:
        raise ValueError("mu_neg_z / mu_pos_z must be (d,) vectors")
    if M.dim() != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"M must be (d, d), got {tuple(M.shape)}")
    d = V.shape[1]
    if mu_neg_z.shape[0] != d or mu_pos_z.shape[0] != d or M.shape[0] != d:
        raise ValueError(f"Monge dims mismatch: d={d}, mu_neg_z {tuple(mu_neg_z.shape)}, M {tuple(M.shape)}")

    z = (pooled_b_c - mu_neg) @ V
    z_tgt = mu_pos_z + (z - mu_neg_z) @ M.T
    delta_z = z_tgt - z
    if task != "add concept":
        delta_z = -delta_z
    delta_z = delta_z * float(strength)
    return delta_z @ V.T


def apply_txt_steering_pooled_advanced(
    pooled_prompt_embeds: torch.Tensor,
    prompt_embeds: torch.Tensor,
    vector_txt: Dict[str, Any],
    normed: bool = True,
    strength: float = 1.0,
    task: str = "add concept",
) -> Tuple[torch.Tensor, torch.Tensor, Union[float, torch.Tensor]]:
    """
    Pooled-only steering using a fixed low-d subspace (mean-diff) or Monge map.
    Sequence branch is zeros; same scale / norm behavior as apply_txt_steering.
    """
    seqs_style = torch.zeros_like(prompt_embeds)
    delta_pooled = compute_pooled_steering_delta_advanced(
        pooled_prompt_embeds, vector_txt, strength=strength, task=task
    )
    return apply_txt_steering(
        pooled_prompt_embeds,
        prompt_embeds,
        delta_pooled,
        seqs_style,
        normed=normed,
        strength=strength,
        task=task,
    )


def apply_txt_steering_auto(
    pooled_prompt_embeds: torch.Tensor,
    prompt_embeds: torch.Tensor,
    vector_txt: Any,
    *,
    strength: float = 1.0,
    task: str = "add concept",
    normed: bool = False,
    legacy_txt_steering: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Union[float, torch.Tensor]]:
    """
    Unified text steering entrypoint.

    - Advanced pooled OT/low-d path when vector_txt["pooled_steering_mode"] in
      {"subspace_mean", "monge"} and legacy_txt_steering is False.
    - Otherwise legacy path (sequence+pooled mean-diff) via steering_txt_data.
    """
    if vector_txt is None:
        zero_seq = torch.zeros_like(prompt_embeds)
        return apply_txt_steering(
            pooled_prompt_embeds,
            prompt_embeds,
            torch.zeros_like(pooled_prompt_embeds),
            zero_seq,
            normed=normed,
            strength=strength,
            task=task,
        )

    mode = pooled_steering_mode(vector_txt)
    if (not legacy_txt_steering) and mode in ("subspace_mean", "monge"):
        return apply_txt_steering_pooled_advanced(
            pooled_prompt_embeds,
            prompt_embeds,
            vector_txt,
            normed=normed,
            strength=strength,
            task=task,
        )

    pooled_style, seqs_style = steering_txt_data(
        vector_txt,
        strength,
        prompt_embeds,
        mean=True,
        ssim=False,
        pooled=True,
        normed=normed,
        task=task,
        seq=False,
    )
    return apply_txt_steering(
        pooled_prompt_embeds,
        prompt_embeds,
        pooled_style,
        seqs_style,
        normed=normed,
        strength=strength,
        task=task,
    )


def apply_txt_steering(pooled_prompt_embeds, prompt_embeds, pooled_style, seqs_style, normed=True, strength=0.0, task='add'):
    #normed = True
    

    if strength != 0.0:
        if task == 'add concept':
            scale = 1
        else:
            scale = F.cosine_similarity(pooled_prompt_embeds.clone() / pooled_prompt_embeds.norm(dim=-1, keepdim=True), pooled_style.clone() / pooled_style.norm(dim=-1, keepdim=True), dim=-1)[0]
            if task != 'add concept':
                scale = -scale
            #assert False

            print('--------------------------------', scale, '-------')
            
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
    mean = False
    seqs_style = vector_txt['sequence'].to(prompt_embeds.dtype).to(prompt_embeds.device)
    if len(seqs_style.shape) == 2:
        seqs_style = seqs_style.unsqueeze(0)

    seqs_style = seqs_style.mean(0, keepdim=True) 
    pooled_style = vector_txt['pooled'].to(prompt_embeds.dtype).to(prompt_embeds.device)
    if len(pooled_style.shape) == 1:
        pooled_style = pooled_style.unsqueeze(0)

    pooled_style = pooled_style.mean(0, keepdim=True) 
    
    
    
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

    if not seq:
        
        seqs_style = torch.zeros_like(seqs_style).to(pooled_style.device).to(pooled_style.dtype)

    if task != 'add concept':
        print('aaaaaaaaaaaaaa')
        pooled_style = -pooled_style
        seqs_style = -seqs_style

    print(pooled_style.shape, seqs_style.sum())
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