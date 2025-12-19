from types import MethodType
from typing import Optional, List
import torch
from diffusers.models.attention_processor import Attention, AttnProcessor
from types import MethodType
from typing import Optional

from diffusers.models.attention_processor import Attention
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import os
from torch import nn
from diffusers.models.activations import GEGLU, GELU, ApproximateGELU, FP32SiLU, LinearActivation, SwiGLU

def get_schedule(timesteps, schedule):
    end = round(len(timesteps) * schedule)
    timesteps = timesteps[:end]
    return timesteps


def calculate_attention_fast(query, key, value, attention_mask=None, scale=None):
    """
    Optimized attention calculation using baddbmm
    
    Args:
        query: (B, H, L, D)
        key: (B, H, S, D)
        value: (B, H, S, D)
        attention_mask: Optional (B, H, L, S) or (B, 1, L, S)
        scale: Optional float, if None will use 1/sqrt(D)
    """
    B, H, L, D = query.shape
    _, _, S, _ = key.shape

    #print(query.shape, key.shape)
    
    # Calculate scaling factor
    if scale is None:
        scale = 1 / math.sqrt(D)
    
    # Reshape inputs to 3D for baddbmm
    query_3d = query.contiguous().view(B * H, L, D)    # (B*H, L, D)
    key_3d = key.contiguous().view(B * H, S, D)        # (B*H, S, D)
    
    # Calculate attention scores
    attention_scores = torch.empty((B * H, L, S), dtype=query.dtype, device=query.device)
    attention_scores = torch.baddbmm(
        torch.empty_like(attention_scores),
        query_3d,                         # (B*H, L, D)
        key_3d.transpose(-2, -1),         # (B*H, D, S)
        beta=0, alpha=scale
    )
    
    # Reshape attention scores back to 4D
    attention_scores = attention_scores.view(B, H, L, S)
    
    # Apply mask if provided
    if attention_mask is not None:
        attention_scores = attention_scores + attention_mask
    
   
    
    return attention_scores


def get_control_config(structure_schedule, appearance_schedule):
    s = structure_schedule
    a = appearance_schedule
    
    control_config =\
f"""control_schedule:
    #       structure_attn
    blocks:                                                # (num layers)
        0: [[{s}]]                                       # 1/0
control_target:
    - [key, query]         # structure_attn   choices: {{query, key, value}}

"""
    
    return control_config

#conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia

def batch_tensor_to_dict(batch_tensor, batch_order):
    batch_tensor_chunk = batch_tensor.chunk(len(batch_order))
    batch_dict = {}
    for i, batch_type in enumerate(batch_order):
        batch_dict[batch_type] = batch_tensor_chunk[i]
    return batch_dict


def batch_dict_to_tensor(batch_dict, batch_order):
    batch_tensor = []
    for batch_type in batch_order:
        batch_tensor.append(batch_dict[batch_type])
    batch_tensor = torch.cat(batch_tensor, dim=0)
    return batch_tensor


def normalize_min_max(features_1, features_2):
    min_val = features_2.min()
    max_val = features_2.max()
    features_1 = (features_1 - features_1.min()) / (features_1.max() - features_1.min())
    features_1 = features_1 * (max_val - min_val) + min_val
    return features_1


def normalize_mean(features_1, features_2):
    mean_val = features_2.mean()
    std_val = features_2.std()

    features_1 = (features_1 - features_1.mean()) / features_1.std()
    features_1 = features_1 * std_val + mean_val
    return features_1



def feature_injection(features, t=0, a=0.5, uncond=True):
    #a = max(0.85, 1 - t / 1000)
    #print(a, t)
    # if uncond:
    #     features[0, :] = features[2, :] * (1-a) + features[0, :] * a
   
    # features[1, :] = features[3, :] * (1-a) + features[1, :] * a

    if features.shape[0] == 6:
        #print('we are here')
        f = 4
    else:
        f = 2

    

    if uncond:
        features[0, :] = features[f, :] * (1-a) + features[0, :] * a
   
    features[1, :] = features[f+1, :] * (1-a) + features[1, :] * a
    
    
    return features


def feature_injection_1(features, t=0, a=0.5, uncond=False):
    #assert False
    #a = max(0.85, 1 - t / 1000)
    #print(a, t)
    # if uncond:
    #     features[0, :] = features[2, :] * (1-a) + features[0, :] * a
   
    # features[1, :] = features[3, :] * (1-a) + features[1, :] * a
    if features.shape == 6:
        f = 4
    else:
        f = 2

    if uncond:
        features[0, :] = features[f, :] * (1-a) + features[0, :] * a
   
    features[1, :] = features[f+1, :] * (1-a) + features[1, :] * a

    
    return features


    
def normalize(x, dim):
    dtype = x.dtype
    x = x.float()
    x_mean = x.mean(dim=dim, keepdim=True)
    x_std = x.std(dim=dim, keepdim=True)
    x_normalized = (x - x_mean) / (x_std + 1e-8)
    x_normalized = x_normalized.to(dtype)
    return x_normalized


def appearance_mean_std(q_c_normed, k_s_normed, v_s):  # c: content, s: style
    q_c = q_c_normed  # q_c and k_s must be projected from normalized features
    k_s = k_s_normed
    mean = F.scaled_dot_product_attention(q_c, k_s, v_s, dropout_p=0.0, is_causal=False)  # Use scaled_dot_product_attention for efficiency
    std = (F.scaled_dot_product_attention(q_c, k_s, v_s.square()) - mean.square()).relu().sqrt()
    
    return mean, std

# def appearance_transfer(features, q_normed, k_normed, batch_order=['uncond', 'cond', 'not_cheaged_uncond', 'not_cheaged_cond', 'photo', 'photo_1'],
#  v=None, reshape_fn=None):
#     assert features.shape[0] % len(batch_order) == 0
#     print("JJJJJJ")

#     features_dict = batch_tensor_to_dict(features, batch_order)
#     q_normed_dict = batch_tensor_to_dict(q_normed, batch_order)
#     k_normed_dict = batch_tensor_to_dict(k_normed, batch_order)
#     v_dict = features_dict
#     if v is not None:
#         v_dict = batch_tensor_to_dict(v, batch_order)
    
#     mean_cond, std_cond = appearance_mean_std(
#         q_normed_dict["not_cheaged_cond"], k_normed_dict["cond"], v_dict["cond"],
#     )

#     if reshape_fn is not None:
#         mean_cond = reshape_fn(mean_cond)
#         std_cond = reshape_fn(std_cond)

#     features_dict["not_cheaged_cond"] = std_cond * normalize(features_dict["not_cheaged_cond"], dim=-2) + mean_cond
    
#     features = batch_dict_to_tensor(features_dict, batch_order)

#     mean_cond, std_cond = appearance_mean_std(
#         q_normed_dict["not_cheaged_uncond"], k_normed_dict["uncond"], v_dict["uncond"],
#     )

#     if reshape_fn is not None:
#         mean_cond = reshape_fn(mean_cond)
#         std_cond = reshape_fn(std_cond)

#     features_dict["not_cheaged_uncond"] = std_cond * normalize(features_dict["not_cheaged_uncond"], dim=-2) + mean_cond
    
#     features = batch_dict_to_tensor(features_dict, batch_order)
#     return features
    

def appearance_transfer(features, q_normed, k_normed, v=None, reshape_fn=None):
    # assert features.shape[0] % len(batch_order) == 0

    
    
    # mean_cond, std_cond = appearance_mean_std(
    #     q_normed[1:2], k_normed[2:3], features[2:3].clone(),
    # )

    # if reshape_fn is not None:
    #     mean_cond = reshape_fn(mean_cond)
    #     std_cond = reshape_fn(std_cond)

    # features[1:2] = std_cond * normalize(features[1:2], dim=-2) + mean_cond

    # mean_cond, std_cond = appearance_mean_std(
    #     q_normed[3:4], k_normed[1:2], features[1:2].clone(),
    # )

    # if reshape_fn is not None:
    #     mean_cond = reshape_fn(mean_cond)
    #     std_cond = reshape_fn(std_cond)

    # features[3:4] = std_cond * normalize(features[3:4], dim=-2) + mean_cond

    mean_cond, std_cond = appearance_mean_std(
        q_normed[3:4], k_normed[1:2], features[1:2].clone(),
    )


    features[3:4] = std_cond * normalize(features[3:4].clone(), dim=-2) + mean_cond
    

    # mean_cond, std_cond = appearance_mean_std(
    #     q_normed[2:3], k_normed[0:1], features[0:1].clone(),
    # )

    # if reshape_fn is not None:
    #     mean_cond = reshape_fn(mean_cond)
    #     std_cond = reshape_fn(std_cond)

    # features[2:3] = std_cond * normalize(features[2:3], dim=-2) + mean_cond
    
    
    # features = batch_dict_to_tensor(features_dict, batch_order)
    return features



class JointAttnProcessor2_Injection:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, do_structure_control=False, do_appearance_control=False, structure_target=None, layer_idx=None, t_threshold=0, block=None):
        
        self.do_structure_control = do_structure_control
        self.do_appearance_control = do_appearance_control
        self.layer_idx = layer_idx
        self.structure_target = structure_target
        self.t_threshold = t_threshold
        self.block = block

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:

        do_appearance_control = self.do_appearance_control
        do_structure_control = self.do_structure_control
        structure_target = self.structure_target
        layer_idx = self.layer_idx

        
        if attn.t >= self.t_threshold:
            do_structure_control  = self.do_structure_control
        else:
            do_structure_control = False

        
        residual = hidden_states

        batch_size = hidden_states.shape[0]

        #do_appearance_control = True
        # if do_appearance_control:
        #     hidden_states_normed = normalize(hidden_states, dim=-2)  # B H D C
        #     hidden_states_normed_1 = normalize(hidden_states_normed, dim=-2)
            
        #     query_normed = attn.to_q(hidden_states_normed)
        #     key_normed = attn.to_k(hidden_states_normed_1)
            
        #     inner_dim = key_normed.shape[-1]
        #     head_dim = inner_dim // attn.heads
        #     query_normed = query_normed.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        #     key_normed = key_normed.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            
        #     # Match query and key injection with structure injection (if injection is happening this layer)
        #     # if do_structure_control:
        #     #     if "query" in attn.structure_target: 
        #     #         query_normed = feature_injection(query_normed, batch_order=attn.batch_order)
        #     #     if "key" in attn.structure_target:
        #     #         key_normed = feature_injection(key_normed, batch_order=attn.batch_order)
        
        # if do_appearance_control:
        #     hidden_states = hidden_states.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        #     hidden_states = appearance_transfer(hidden_states, query_normed, key_normed)
        #     hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        # if do_appearance_control:
            
        #     hidden_states_normed = normalize(hidden_states, dim=-2)  # B H D C
        #     hidden_states_normed_1 = normalize(hidden_states_normed, dim=-2)
            
        #     query_normed = attn.to_q(hidden_states_normed)
        #     key_normed = attn.to_k(hidden_states_normed_1)
            
        #     inner_dim = key_normed.shape[-1]
        #     head_dim = inner_dim // attn.heads
        #     query_normed = query_normed.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        #     key_normed = key_normed.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            
        #     # Match query and key injection with structure injection (if injection is happening this layer)
        #     # if do_structure_control:
        #     #     if "query" in attn.structure_target: 
        #     #         query_normed = feature_injection(query_normed, batch_order=attn.batch_order)
        #     #     if "key" in attn.structure_target:
        #     #         key_normed = feature_injection(key_normed, batch_order=attn.batch_order)
        
        # if do_appearance_control:
        #     #print('do control')
        #     hidden_states = hidden_states.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        #     hidden_states = appearance_transfer(hidden_states, query_normed, key_normed)
        #     hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
         
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        #print(do_structure_control)
        if do_structure_control:
            #assert False
            print(attn.t, attn.a)
            
            if "query" in structure_target: 
                query = feature_injection(query, a=attn.a, t=attn.t)
            if "key" in structure_target:
                key = feature_injection(key, a=attn.a, t=attn.t)
            if "value" in structure_target:
                value = feature_injection(value, a=attn.a, t=attn.t)
        
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

        
        #print(query.shape, query.shape[2] == 4429, self.block)
        if True:
            # if attn.t > 500 and self.block is not None and query.shape[2] == 4429:
            #     os.makedirs('activations_test_hat_bad', exist_ok=True)
            #     print(attn.t, self.block)
            #     torch.save({'query': query, 'key': key}, f'activations_test_hat_bad/key_query_{attn.t}_{self.block}')
               
            hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        else:
            
            B, H, L, D = query.shape
            _, _, S, _ = key.shape
            if encoder_hidden_states is not None:
                image_length = query.shape[2] - encoder_hidden_states_query_proj.shape[2]
            else:
                image_length = query.shape[2]

            temperature = 1
            attention_scores = calculate_attention_fast(query, key, value)

            #attention_probs_txt = attention_scores[:, :, image_length:, image_length:].cpu()
            attention_probs = F.softmax(attention_scores / temperature, dim=-1)
            
            if do_structure_control:
                # features_dict = batch_tensor_to_dict(attention_probs, attn.batch_order)
                # summ = features_dict["cond"][:, :, :image_length, :image_length].float().sum(axis=-1, keepdim=True)
                # summ1 = features_dict["structure_cond"][:, :, :image_length, :image_length].float().sum(axis=-1, keepdim=True)
                # features_dict["cond"][:, :, :image_length, :image_length] = features_dict["structure_cond"][:, :, :image_length, :image_length].float() * (summ + 1e-8) / (summ1 + 1e-8)
                # features_dict["cond"][:, :, :image_length, :image_length] = features_dict["cond"][:, :, :image_length, :image_length].to(features_dict["structure_cond"].dtype)

                a = 0.6
                if value.shape[0] == 6:
                    f = 4
                else:
                    f = 2

                
                attention_probs[1, :, :image_length, :image_length] = attention_probs[f+1, :, :image_length, :image_length]#.float()
                attention_probs[0, :, :image_length, :image_length] = attention_probs[f, :, :image_length, :image_length]#.float() #* (1-a) + features_dict["uncond"][:, :, :image_length, :image_length]*a
                 
                attention_probs = attention_probs  / (attention_probs.sum(axis=-1, keepdim=True) + 1e-8)
                attention_probs = attention_probs.to(attention_scores.dtype)
            #attention_probs = F.softmax(attention_scores / temperature, dim=-1)
                
            #attention_probs = F.softmax(attention_scores / temperature, dim=-1)
            # Reshape for final multiplication
            attention_probs_3d = attention_probs.view(B * H, L, S)
            
            # Calculate hidden states
            value_3d = value.contiguous().view(B * H, S, D)    # (B*H, S, D)
            hidden_states = torch.bmm(attention_probs_3d, value_3d)  # (B*H, L, D)
            
            # Reshape output back to 4D
            hidden_states = rearrange(hidden_states, '(b h) c s -> b h c s', b=batch_size)
            #hidden_states = torch.dropout(attn_weight, dropout_p, train=True) @ value

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            # Split the attention outputs.
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        # if do_appearance_control:
            
        #     hidden_states_normed = normalize(hidden_states.clone(), dim=-2)  # B H D C
        #     #hidden_states_normed_1 = normalize(hidden_states_normed, dim=-2)
            
        #     query_normed = attn.to_q(hidden_states_normed.clone())
        #     key_normed = attn.to_k(hidden_states_normed.clone())
            
        #     inner_dim = key_normed.shape[-1]
        #     head_dim = inner_dim // attn.heads
        #     query_normed = query_normed.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        #     key_normed = key_normed.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            
            # Match query and key injection with structure injection (if injection is happening this layer)
            # if do_structure_control:
            #     if "query" in attn.structure_target: 
            #         query_normed = feature_injection(query_normed, batch_order=attn.batch_order)
            #     if "key" in attn.structure_target:
            #         key_normed = feature_injection(key_normed, batch_order=attn.batch_order)
        
        # if do_appearance_control:
        #     #print('do control')
        #     hidden_states = hidden_states.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        #     hidden_states = appearance_transfer(hidden_states, query_normed, key_normed)
        #     hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

      

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states

