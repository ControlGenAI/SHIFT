# utils.py

import torch
import numpy as np
import math
from typing import Dict, Any, Optional, List, Tuple, Union
import os

import sklearn.svm._classes

def orthogonal_projection_steering(
    current_output: torch.Tensor, 
    steering_tensor: torch.Tensor, 
    normalize: bool = True
) -> torch.Tensor:
    """
    Calculates the component of the steering_tensor orthogonal to the current_output.
    
    This function handles the common case where both tensors represent [L, D] or [B, L, D] 
    data, or where steering_tensor [D] can be broadcast.

    Args:
        current_output: The activation tensor (e.g., token embeddings).
        steering_tensor: The steering vector tensor (can be [D], [L, D], or [B, L, D]).
        normalize: If True, normalize the final orthogonal component to unit length.

    Returns:
        The orthogonal steering tensor, with the same shape as current_output/steering_tensor.
    """
    # 1. Standardize and check shapes
    original_dtype = current_output.dtype
    output_f = current_output.float()
    steering_f = steering_tensor.float()

    # Normalize the current output (the basis vector)
    norm_basis = torch.norm(output_f, dim=-1, keepdim=True) + 1e-6
    normalized_output = output_f / norm_basis

    norm_steering = torch.norm(steering_f, dim=-1, keepdim=True) + 1e-6
    normalized_steering = steering_f / norm_steering # Normalized direction vector

    # --- Projection Calculation (Project normalized_steering onto normalized_output) ---
    final_orth_steering = []

    if False: #len(normalized_steering) != 1:
        for i in range(len(normalized_steering)):
            dot_product_numerator = torch.sum(normalized_steering[i] * normalized_output, dim=-1, keepdim=True)

            projection_vector = dot_product_numerator * normalized_output

            orthogonal_steering = normalized_steering[i] - projection_vector

            # just check orthogonality, may be deleted in the future
            orthogonality_check = torch.sum(orthogonal_steering * normalized_output, dim=-1).abs().max()
            assert orthogonality_check < 1e-6, f"Orthogonal component is NOT orthogonal. Max dot product magnitude: {orthogonality_check.item()}"

            if normalize:
                # Normalize the orthogonal component itself (to make it a unit direction vector)
                ortho_norm = torch.norm(orthogonal_steering, dim=-1, keepdim=True) + 1e-6
                orthogonal_steering = orthogonal_steering / ortho_norm
            final_orth_steering.append(orthogonal_steering)
        orthogonal_steering = torch.stack(final_orth_steering)
        del normalized_steering
        del final_orth_steering
    else:
        dot_product_numerator = torch.sum(normalized_steering * normalized_output, dim=-1, keepdim=True)

        projection_vector = dot_product_numerator * normalized_output

        orthogonal_steering = normalized_steering - projection_vector

        # just check orthogonality, may be deleted in the future
       
        if normalize:
            # Normalize the orthogonal component itself (to make it a unit direction vector)
            ortho_norm = torch.norm(orthogonal_steering, dim=-1, keepdim=True) + 1e-6
            orthogonal_steering = orthogonal_steering / ortho_norm

        orthogonality_check = torch.sum(orthogonal_steering * normalized_output, dim=-1).abs().max()
        assert orthogonality_check < 1e-5, f"Orthogonal component is NOT orthogonal. Max dot product magnitude: {orthogonality_check.item()}"

        # if normalize:
        #     # Normalize the orthogonal component itself (to make it a unit direction vector)
        #     ortho_norm = torch.norm(orthogonal_steering, dim=-1, keepdim=True) + 1e-6
        #     orthogonal_steering = orthogonal_steering / ortho_norm

    return orthogonal_steering.to(original_dtype)


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
    use_distance: bool = True,
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
    print(cls_min)
    if not use_distance:
        score_cls = model.predict_proba(a) if model is not None else np.array([[0.5, 0.5]])
        if task == 'add':
            scoreeee =  21 * score_cls[0][1]**2   #min(cls_min, (1 / ((1 - score_cls[0][0]) + 1e-8) - 1))
        else:
            #assert False
            scoreeee = (1 / ((1 - score_cls[0][1]) + 1e-8) - 1).clip(0, cls_min) #  21 * score_cls[0][1]**2  
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


def load_steering_data(
    svm_model_path: Optional[str], 
    scores_path: Optional[str], 
    token_best_path: Optional[str], 
    quantile_level: float
) -> Tuple[Optional[Dict], Optional[torch.Tensor], Optional[torch.Tensor], Optional[np.ndarray]]:
    """Loads SVM models, best tokens, and scores."""
    
    models = None
    if svm_model_path and os.path.exists(svm_model_path):
        with torch.serialization.safe_globals([sklearn.svm._classes.SVC]): 
            # Note: weights_only=False must be used for loading custom objects like SVMs
            models = torch.load(svm_model_path, weights_only=False) 

    tokens_best = None
    if token_best_path and os.path.exists(token_best_path):
        tokens_best = torch.load(token_best_path, weights_only=False)
     
    scores_all = None
    if scores_path and os.path.exists(scores_path):
        # Scores are typically NumPy arrays, load them and convert to torch/numpy for quantile
        scores_all = torch.from_numpy(torch.load(scores_path, weights_only=False))

    quantiles = None
    if scores_all is not None:
        scores_np = scores_all.numpy()
        if 'block' in scores_path: # Infer from filename or use a separate argument
            # Quantiles calculated over tokens, resulting in (Timestep x Layer)
            quantiles = np.quantile(scores_np, q=quantile_level, axis=2) 
        elif 'timestep' in scores_path:
            # Quantiles calculated over layers and tokens, resulting in (Timestep)
            quantiles = np.quantile(scores_np, q=quantile_level, axis=(1, 2)) 
        else: # Overall quantile
            quantiles = np.quantile(scores_np, q=quantile_level)
    
    return models, tokens_best, scores_all, quantiles