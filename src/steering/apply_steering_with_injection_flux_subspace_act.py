"""
Inference with Subspace-AcT steering for FLUX/SD3 dual branches.

Expected vector format (from calculate_steering_vector_subspace_act.py):
  vector[step][layer] = {
    "img": {"method", "U", "mu_neg", "cpca_tokenwise"?, ...},
    "txt": {"method", "U", "mu_neg", "cpca_tokenwise"?, ...},
  }
  If cpca_tokenwise is true, steering uses Z = (A - mu_neg) U per token; else one z from mean-pooled A.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import sklearn.svm._classes
import torch
from datasets import load_dataset

STEERING_ROOT = Path(__file__).resolve().parents[2]
if str(STEERING_ROOT) not in sys.path:
    sys.path.insert(0, str(STEERING_ROOT))

from src.models.flux import FluxPipeline
from src.models.sd_3 import StableDiffusion3Pipeline
from src.utils.utils import calculate_cls_score


MODEL_CONFIGS = {
    "flux": {"last_layer": 56, "out_idx": 1, "inner_idx": 0, "dtype": torch.bfloat16},
    "sd3": {"last_layer": 23, "out_idx": 1, "inner_idx": 1, "dtype": torch.float16},
}


def load_steering_data(
    svm_model_path: Optional[str],
    scores_path: Optional[str],
    token_best_path: Optional[str],
    quantile_level: float,
) -> Tuple[Optional[Dict], Optional[torch.Tensor], Optional[torch.Tensor], Optional[np.ndarray]]:
    models = None
    if svm_model_path and os.path.exists(svm_model_path):
        with torch.serialization.safe_globals([sklearn.svm._classes.SVC]):
            models = torch.load(svm_model_path, weights_only=False)

    tokens_best = None
    if token_best_path and os.path.exists(token_best_path):
        tokens_best = torch.load(token_best_path, weights_only=False)

    scores_all = None
    if scores_path and os.path.exists(scores_path):
        data = torch.load(scores_path, weights_only=False)
        scores_all = data if torch.is_tensor(data) else torch.from_numpy(data)

    quantiles = None
    if scores_all is not None and scores_path:
        scores_np = scores_all.numpy()
        if "block" in scores_path:
            quantiles = np.quantile(scores_np, q=quantile_level, axis=2)
        elif "timestep" in scores_path:
            quantiles = np.quantile(scores_np, q=quantile_level, axis=(1, 2))
        else:
            quantiles = np.quantile(scores_np, q=quantile_level)
    return models, tokens_best, scores_all, quantiles


def split_vector(vector: dict) -> Tuple[dict, dict]:
    """Split dual-stream vector into img/txt dicts. Legacy falls back to txt-only."""
    img_vec = {}
    txt_vec = {}

    is_dual = False
    for step in vector:
        if not isinstance(step, int):
            continue
        for layer_key in vector[step]:
            val = vector[step][layer_key]
            if isinstance(val, dict) and ("img" in val or "txt" in val):
                is_dual = True
            break
        break

    if not is_dual:
        return {}, vector

    for step in vector:
        if not isinstance(step, int):
            continue
        img_vec[step] = {}
        txt_vec[step] = {}
        for layer_key in vector[step]:
            val = vector[step][layer_key]
            if not isinstance(val, dict):
                txt_vec[step][layer_key] = val
                continue
            if "img" in val:
                img_vec[step][layer_key] = val["img"]
            if "txt" in val:
                txt_vec[step][layer_key] = val["txt"]

    return img_vec, txt_vec


def _has_any_layer_payload(stream_vec: dict) -> bool:
    """True only if at least one step contains at least one layer payload."""
    for step, layers in stream_vec.items():
        if not isinstance(step, int):
            continue
        if isinstance(layers, dict) and len(layers) > 0:
            return True
    return False


def _payload_is_tokenwise(payload: dict) -> bool:
    """True if vector was built with --cpca_tokenwise (per-token Z and ΔA)."""
    v = payload.get("cpca_tokenwise")
    if v is None:
        return False
    if torch.is_tensor(v):
        return bool(v.item())
    return bool(v)


def _transport_subspace(z: torch.Tensor, payload: dict) -> torch.Tensor:
    method = payload.get("method", "")
    if method == "subspace_affine":
        W = payload["W"].to(z.device).float()  # float32, не z.dtype
        b = payload["b"].to(z.device).float()
        # Pooled payload: W (d, d), b (d,). Tokenwise per-token fit: W (T, d, d), b (T, d).
        if W.dim() == 2:
            return z @ W.T + b
        T_train = W.shape[0]
        N = z.shape[0]
        if N == T_train:
            return torch.einsum("td,tdk->tk", z, W) + b
        if N % T_train == 0:
            zv = z.view(-1, T_train, z.shape[-1])
            out = torch.einsum("btd,tdk->btk", zv, W) + b.unsqueeze(0)
            return out.reshape(N, z.shape[-1])
        Wm = W.mean(dim=0)
        bm = b.mean(dim=0)
        return z @ Wm.T + bm
    if method == "subspace_meandiff":
        delta_z = payload["delta_z"].to(z.device).float()
        # Per-token meandiff: delta_z has shape (T_train, d).
        # z is (N, d) where N = B * T_infer. Broadcast per-token if T aligns.
        if delta_z.dim() == 2:
            T_train = delta_z.shape[0]
            N = z.shape[0]
            if N == T_train:
                return z + delta_z
            if N % T_train == 0:
                B = N // T_train
                return z + delta_z.repeat(B, 1)
            # T mismatch — fall back to average per-token shift so at least something
            # is applied; warn once via payload flag if needed.
            return z + delta_z.mean(dim=0, keepdim=True)
        return z + delta_z
    if method == "subspace_sinkhorn":
        z_pos = payload["z_pos"].to(z.device).float()
        z_in = z if z.dim() == 2 else z.unsqueeze(0)     # (Q, d)
        # Preferred: continuous barycentric extension via target dual potential.
        if "g" in payload and "epsilon" in payload:
            g_pot = payload["g"].to(z.device).float()    # (M,)
            eps_t = payload["epsilon"]
            eps = float(eps_t.item()) if torch.is_tensor(eps_t) else float(eps_t)
            sq_dist = torch.cdist(z_in.float(), z_pos, p=2).pow(2)   # (Q, M)
            log_w = g_pot.unsqueeze(0) - sq_dist / eps               # (Q, M)
            weights = torch.softmax(log_w, dim=1)
            z_out = weights @ z_pos
            return z_out if z.dim() == 2 else z_out.squeeze(0)
        # Backward-compat: legacy payloads stored z_neg+gamma (NN barycenter).
        if "z_neg" in payload and "gamma" in payload:
            z_neg = payload["z_neg"].to(z.device).float()
            gamma = payload["gamma"].to(z.device).float()
            dists = torch.cdist(z_in.float(), z_neg, p=2)
            nn_idx = torch.argmin(dists, dim=1)
            gamma_rows = gamma[nn_idx]
            weights = gamma_rows / gamma_rows.sum(dim=1, keepdim=True).clamp(min=1e-8)
            z_out = weights @ z_pos
            return z_out if z.dim() == 2 else z_out.squeeze(0)
        raise KeyError(
            "subspace_sinkhorn payload must contain either ('g','epsilon') "
            "or legacy ('z_neg','gamma'). Re-run calculation with the new format."
        )
    raise ValueError(f"Unsupported Subspace-AcT method: {method}")


# def _apply_subspace_steering(
#     to_modify: torch.Tensor,
#     payload: dict,
#     alpha: float,
#     transport_lambda: float,
#     score_val: torch.Tensor,
# ) -> torch.Tensor:
#     """
#     Implements:
#       z  = U^T(a - mu_neg)
#       z' = (1-lambda) z + lambda T(z)
#       da = U (z' - z)
#       a~ = a + alpha * score * da
#     where a is pooled current activation.
#     """
#     orig_dtype = to_modify.dtype
#     compute_dtype = torch.float32

#     U = payload["U"].to(to_modify.device, compute_dtype)            # (C, d)
#     mu_neg = payload["mu_neg"].to(to_modify.device, compute_dtype)  # (C,)

#     x = to_modify.to(compute_dtype)
#     if x.dim() == 2:
#         # (T, C) -> treat as single batch item
#         a_pool = x.mean(dim=0, keepdim=True)   # (1, C)
#         squeeze_batch = True
#     elif x.dim() == 3:
#         # (B, T, C)
#         a_pool = x.mean(dim=1)                 # (B, C)
#         squeeze_batch = False
#     else:
#         # Fallback for unexpected rank: flatten all non-feature dims into a single "token" axis.
#         x = x.reshape(1, -1, x.shape[-1])
#         a_pool = x.mean(dim=1)
#         squeeze_batch = False

#     z = (a_pool - mu_neg) @ U                  # (B, d)
#     z_t = _transport_subspace(z, payload)      # (B, d)
#     z_prime = (1.0 - transport_lambda) * z + transport_lambda * z_t
#     delta_a = (z_prime - z) @ U.T              # (B, C)

#     # Optional dynamic modulation via classifier score
#     batch_size = int(a_pool.shape[0])
#     score_flat = score_val.float().reshape(-1).to(to_modify.device, compute_dtype)
#     if score_flat.numel() == 1:
#         score_batch = score_flat.expand(batch_size)
#     elif score_flat.numel() == batch_size:
#         score_batch = score_flat
#     else:
#         # Fallback to scalar when classifier output shape does not match current batch.
#         score_batch = score_flat.mean().expand(batch_size)
#     eff_alpha = alpha * score_batch
#     if squeeze_batch:
#         delta_view = delta_a.squeeze(0).unsqueeze(0)     # (1, C) for (T, C)
#         out = x + eff_alpha[0] * delta_view
#     else:
#         delta_view = delta_a.unsqueeze(1)                # (B, 1, C) for (B, T, C)
#         out = x + eff_alpha.view(-1, 1, 1) * delta_view
#     return out.to(orig_dtype)

def _calculate_concept_subspace_ratio(to_modify, payload):
    U = payload["U"].to(to_modify.device).float()       # (C, d)
    mu_neg = payload["mu_neg"].to(to_modify.device).float()  # (C,)
    
    flat = to_modify.reshape(-1, to_modify.shape[-1]).float()  # (T, C)
    centered = flat - mu_neg                                    # (T, C)
    
    # Проекция на концептное подпространство
    z = centered @ U                                    # (T, d)
    proj = z @ U.T                                      # (T, C) — компонента в span(U)
    
    # Норма проекции и норма исходного вектора
    proj_norm = proj.norm(dim=-1)                       # (T,)
    total_norm = centered.norm(dim=-1).clamp(min=1e-8)  # (T,)
    
    # Score для каждого токена
    token_scores = proj_norm / total_norm               # (T,) — от 0 до 1
    
    # Агрегируем по токенам — среднее
    score = token_scores.mean()                         # скаляр
    
    return score, token_scores                          # скаляр + (T,) для детального анализа

def _apply_subspace_steering(
    to_modify,
    payload,
    alpha,
    transport_lambda,
    score_val,
    energy_restoration=False,
    meandiff_neutral_k: int = 0,
):
    # Держим всю математику в float32
    U = payload["U"].to(to_modify.device).float()
    mu_neg = payload["mu_neg"].to(to_modify.device).float()

    orig_shape = to_modify.shape
    flat = to_modify.reshape(-1, to_modify.shape[-1]).float()

    if _payload_is_tokenwise(payload):
        # Z = (A - mu_neg) U per token row; ΔA = (Z' - Z) U^T (same as matrix form).
        z = (flat - mu_neg) @ U
        z_t = _transport_subspace(z, payload)
        z_prime = (1.0 - transport_lambda) * z + transport_lambda * z_t
        delta_flat = (z_prime - z) @ U.T
    else:
        a = flat.mean(dim=0)
        z = (a - mu_neg) @ U
        z_t = _transport_subspace(z, payload)
        z_prime = (1.0 - transport_lambda) * z + transport_lambda * z_t
        delta_vec = (z_prime - z) @ U.T
        delta_flat = delta_vec.unsqueeze(0).expand_as(flat)

    if (
        meandiff_neutral_k > 0
        and payload.get("method") == "subspace_meandiff"
        and flat.shape[0] >= 2
    ):
        assert False
        act_centered = flat #- flat.mean(dim=0, keepdim=True)
        _, _, v_t = torch.linalg.svd(act_centered, full_matrices=False)
        k_eff = min(
            int(meandiff_neutral_k),
            int(v_t.shape[0]),
            int(flat.shape[-1]),
        )
        if k_eff > 0:
            u_neutral = v_t[:k_eff].T.contiguous()
            delta_flat = delta_flat - (delta_flat @ u_neutral) @ u_neutral.T

    score, token_scores = _calculate_concept_subspace_ratio(to_modify, payload)
    #print(f"score: {score}", token_scores.shape)

    score_scalar = score_val.float().mean()
    print(score_val)
    eff_alpha = alpha * score_scalar

    if args.task == 'remove' or args.task == 'nudity':
        eff_alpha = -eff_alpha

    steered = to_modify.float() + eff_alpha * delta_flat.reshape(orig_shape)

    if energy_restoration:
        # Restore per-token activation norm to reduce amplitude drift after steering.
        eps = 1e-6
        orig_norm = to_modify.float().norm(dim=-1, keepdim=True).clamp(min=eps)
        steered_norm = steered.norm(dim=-1, keepdim=True).clamp(min=eps)
        steered = (steered / steered_norm) * orig_norm

    return steered.to(to_modify.dtype)  # кастим только результат


def _pack_activation_archive_like_get_vector(baseline_acts: dict) -> dict:
    """
    Same top-level layout as get_vector_1 / activation .pt files:
      {step: {layer_N: {"txt"|"img": tensor (N,T,C)}}}
    int step keys only (plus optional __dump_meta__ added by caller).
    """
    out: Dict = {}
    for step, layers in baseline_acts.items():
        step_i = int(step)
        out[step_i] = {}
        for layer_key, branches in layers.items():
            out[step_i][layer_key] = {}
            for br, t in branches.items():
                x = t
                if x.dim() == 2:
                    x = x.unsqueeze(0)
                if x.dim() != 3:
                    raise ValueError(
                        f"Expected (N,T,C) or (T,C), got {tuple(x.shape)} "
                        f"for step={step_i} {layer_key} {br}"
                    )
                out[step_i][layer_key][br] = x
    return out


def _maybe_record_baseline_activation(
    hook_state: dict,
    args,
    step: int,
    layer_idx: int,
    branch: str,
    tensor: torch.Tensor,
) -> None:
    """Накопление сырых активаций до стиринга; torch.save снаружи (см. --dump_baseline_activations)."""
    path = hook_state.get("__dump_baseline_path__")
    if not path:
        return
    if hook_state.get("__current_prompt_idx__", -1) != hook_state.get(
        "__dump_baseline_prompt_idx__", 0
    ):
        return
    if branch == "txt" and float(args.strength) != 0.0:
        return
    if branch == "img" and float(args.strength_img) != 0.0:
        return
    layer_key = f"layer_{layer_idx}"
    acts = hook_state.setdefault("__baseline_acts__", {})
    acts.setdefault(int(step), {}).setdefault(layer_key, {})[branch] = (
        tensor.detach().cpu().float()
    )


def _unwrap_branch_hidden(hidden, inner_idx: int):
    """
    Keep full tensor when hidden is already a Tensor.
    Use inner index only for nested tuple/list outputs.
    """
    if torch.is_tensor(hidden):
        return hidden.clone(), False
    if isinstance(hidden, (tuple, list)):
        if len(hidden) > inner_idx and torch.is_tensor(hidden[inner_idx]):
            return hidden[inner_idx].clone(), True
    raise TypeError(f"Unsupported branch hidden type: {type(hidden)}")


def apply_attention_steering(pipe, args, vector):
    m_key = "flux" if "flux" in args.model_name.lower() else "sd3"
    cfg = MODEL_CONFIGS[m_key]
    state = {"step": 0}
    hook_handles = []

    img_vector, txt_vector = split_vector(vector)
    has_img = _has_any_layer_payload(img_vector)
    has_txt = _has_any_layer_payload(txt_vector)
    print(f"  Steering branches: img={'YES' if has_img else 'NO'}, txt={'YES' if has_txt else 'NO'}")

    eigen_path = os.path.join(args.data_dir, "cos_sep.pt")
    svm_path = os.path.join(args.data_dir, "base_0.85_135_svm_models.pt")
    scr_path = os.path.join(args.data_dir, "base_0.85_135_scores.pt")
    tok_path = os.path.join(args.data_dir, "base_0.85_135_tokens.pt")

    models, tokens_best, scores_all, quantiles = load_steering_data(
        svm_path, scr_path, tok_path, args.quantile_level
    )

    try:
        eigen_info = torch.load(eigen_path, weights_only=False)
    except Exception:
        eigen_info = None

    def _get_score_val(step, layer_idx, to_modify, models, scores_all, eigen_info, cfg, args):
        score_val = torch.ones((1, 1), device=to_modify.device, dtype=to_modify.dtype)
        current_signal = 1.0
        if scores_all is not None:
            layer_scores = scores_all[:, step, layer_idx]
            current_signal = layer_scores.float()

        if args.use_cls and models:
            ensemble = models.get(step, {}).get(f"layer_{layer_idx}")
            if ensemble:
                # Classifier expects a single feature vector [1, C].
                # to_modify can be (T, C) or (B, T, C), so pool all token/batch rows.
                mean_act = to_modify.float().reshape(-1, to_modify.shape[-1]).mean(0, keepdim=True)
                mean_act = mean_act / (mean_act.norm(dim=-1, keepdim=True) + 1e-6)
                votes = [
                    calculate_cls_score(
                        mean_act.cpu().float(), args.cls_min, m,
                        args.cls_type, task=args.task, use_distance=False
                    )[0]
                    if current_signal[i] > args.min_signal_threshold else 1.0
                    for i, m in enumerate(ensemble)
                ]
                print(votes)
                # False    
                score_val = torch.tensor(votes).to(to_modify.device, to_modify.dtype)
                score_val = torch.mean(score_val, dim=0, keepdim=True)
        return score_val

    def _get_layer_payloads(step: int, layer_idx: int):
        layer_key = f"layer_{layer_idx}"
        p_txt = txt_vector.get(step, {}).get(layer_key)
        p_img = img_vector.get(step, {}).get(layer_key)
        return p_img, p_txt

    def steering_hook(layer_idx: int):
        def hook(module, input, output):
            step = state["step"]
            if args.block_steering != "all" and layer_idx not in args.block_steering:
                return output
            if args.t_steering != "all" and step not in args.t_steering:
                return output

            print(f"step: {step}, layer_idx: {layer_idx}")
            payload_img, payload_txt = _get_layer_payloads(step, layer_idx)
            if payload_img is None and payload_txt is None:
                return output

            if args.injection_point == "attn":
                if not (isinstance(output, tuple) and len(output) == 2):
                    return output
                act_tuple = list(output)

                if payload_txt is not None:
                    txt_hidden = act_tuple[1]
                    try:
                        txt_to_modify, use_inner_txt = _unwrap_branch_hidden(
                            txt_hidden, cfg["inner_idx"]
                        )
                    except TypeError:
                        return output
                    _maybe_record_baseline_activation(
                        state, args, step, layer_idx, "txt", txt_to_modify
                    )
                    score_val = _get_score_val(step, layer_idx, txt_to_modify, models, scores_all, eigen_info, cfg, args)
                    steered_txt = _apply_subspace_steering(
                        txt_to_modify,
                        payload_txt,
                        alpha=args.strength,
                        transport_lambda=args.transport_lambda_txt,
                        score_val=score_val,
                        energy_restoration=args.energy_restoration,
                        meandiff_neutral_k=args.meandiff_neutral_k,
                    )
                    if use_inner_txt:
                        act_tuple[1][cfg["inner_idx"]] = steered_txt
                    else:
                        act_tuple[1] = steered_txt

                if payload_img is not None:
                    img_hidden = act_tuple[0]
                    try:
                        img_to_modify, use_inner_img = _unwrap_branch_hidden(
                            img_hidden, cfg["inner_idx"]
                        )
                    except TypeError:
                        return output
                    _maybe_record_baseline_activation(
                        state, args, step, layer_idx, "img", img_to_modify
                    )
                    score_val = _get_score_val(step, layer_idx, img_to_modify, models, scores_all, eigen_info, cfg, args)
                    steered_img = _apply_subspace_steering(
                        img_to_modify,
                        payload_img,
                        alpha=args.strength_img,
                        transport_lambda=args.transport_lambda_img,
                        score_val=score_val,
                        energy_restoration=args.energy_restoration,
                        meandiff_neutral_k=args.meandiff_neutral_k,
                    )
                    if use_inner_img:
                        act_tuple[0][cfg["inner_idx"]] = steered_img
                    else:
                        act_tuple[0] = steered_img

                return tuple(act_tuple)

            if args.injection_point == "block":
                if not (isinstance(output, tuple) and len(output) == 2):
                    return output
                txt_hidden, img_hidden = output[0], output[1]
                txt_new, img_new = txt_hidden, img_hidden

                if payload_txt is not None:
                    txt_to_modify = txt_hidden.clone()
                    _maybe_record_baseline_activation(
                        state, args, step, layer_idx, "txt", txt_to_modify
                    )
                    score_val = _get_score_val(step, layer_idx, txt_to_modify, models, scores_all, eigen_info, cfg, args)
                    print(score_val)
                    #assert False
                    txt_new = _apply_subspace_steering(
                        txt_to_modify,
                        payload_txt,
                        alpha=args.strength,
                        transport_lambda=args.transport_lambda_txt,
                        score_val=score_val,
                        energy_restoration=args.energy_restoration,
                        meandiff_neutral_k=args.meandiff_neutral_k,
                    )
                if payload_img is not None:
                    img_to_modify = img_hidden.clone()
                    _maybe_record_baseline_activation(
                        state, args, step, layer_idx, "img", img_to_modify
                    )
                    score_val = _get_score_val(step, layer_idx, img_to_modify, models, scores_all, eigen_info, cfg, args)
                    img_new = _apply_subspace_steering(
                        img_to_modify,
                        payload_img,
                        alpha=args.strength_img,
                        transport_lambda=args.transport_lambda_img,
                        score_val=score_val,
                        energy_restoration=args.energy_restoration,
                        meandiff_neutral_k=args.meandiff_neutral_k,
                    )
                return (txt_new, img_new)

            return output
        return hook

    def steering_hook_ff(layer_idx: int, branch: str):
        def hook(module, input, output):
            step = state["step"]
            if args.block_steering != "all" and layer_idx not in args.block_steering:
                return output
            if args.t_steering != "all" and step not in args.t_steering:
                return output
            payload_img, payload_txt = _get_layer_payloads(step, layer_idx)
            payload = payload_img if branch == "img" else payload_txt
            if payload is None or not torch.is_tensor(output):
                return output
            to_modify = output.clone()
            _maybe_record_baseline_activation(state, args, step, layer_idx, branch, to_modify)
            score_val = _get_score_val(step, layer_idx, to_modify, models, scores_all, eigen_info, cfg, args)
            alpha = args.strength_img if branch == "img" else args.strength
            lam = args.transport_lambda_img if branch == "img" else args.transport_lambda_txt
            return _apply_subspace_steering(
                to_modify,
                payload,
                alpha=alpha,
                transport_lambda=lam,
                score_val=score_val,
                energy_restoration=args.energy_restoration,
                meandiff_neutral_k=args.meandiff_neutral_k,
            )
        return hook

    if args.injection_point in ("attn", "block", "ff") and hasattr(pipe.transformer, "transformer_blocks"):
        blocks = pipe.transformer.transformer_blocks
        for layer_id, block in enumerate(blocks):
            if args.injection_point == "attn":
                hook_handles.append(block.attn.register_forward_hook(steering_hook(layer_id)))
            elif args.injection_point == "block":
                hook_handles.append(block.register_forward_hook(steering_hook(layer_id)))
            elif args.injection_point == "ff":
                hook_handles.append(block.ff.register_forward_hook(steering_hook_ff(layer_id, "img")))
                hook_handles.append(block.ff_context.register_forward_hook(steering_hook_ff(layer_id, "txt")))
    else:
        layer_id = 0
        for name, module in pipe.transformer.named_modules():
            if name.endswith("attn"):
                hook_handles.append(module.register_forward_hook(steering_hook(layer_id)))
                layer_id += 1

    return state, lambda: [h.remove() for h in hook_handles]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="black-forest-labs/FLUX.1-schnell")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--prompts_path", type=str, default="data/captions.txt")
    parser.add_argument("--vector_type", type=str, default="subspace_act_v1")
    parser.add_argument("--num_prompts", type=int, default=500)
    parser.add_argument("--task", type=str, default="add concept", choices=["add concept", "remove", "nudity"])
    parser.add_argument("--remove_prompt", type=str, default="cyberpunk style")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)

    parser.add_argument("--strength", type=float, default=20.0, help="Text branch alpha")
    parser.add_argument("--strength_img", type=float, default=20.0, help="Image branch alpha")
    parser.add_argument("--transport_lambda_txt", type=float, default=1.0, help="Lambda in z'=(1-l)z+lT(z) for txt")
    parser.add_argument("--transport_lambda_img", type=float, default=1.0, help="Lambda in z'=(1-l)z+lT(z) for img")
    parser.add_argument(
        "--energy_restoration",
        action="store_true",
        help="Restore original activation norm after steering update.",
    )
    parser.add_argument(
        "--injection_point",
        type=str,
        default="attn",
        choices=["attn", "block", "ff"],
    )
    parser.add_argument("--use_cls", action="store_true")
    parser.add_argument("--min_signal_threshold", type=float, default=0.5)

    parser.add_argument("--block_steering", type=str, default="all")
    parser.add_argument("--t_steering", type=str, default="all")
    parser.add_argument("--quantile_level", type=float, default=0.5)

    parser.add_argument("--steer_txt", action="store_true")
    parser.add_argument("--strength_txt", type=float, default=0.0)
    parser.add_argument("--inference_steps", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_dir", type=str, default="results_steered")
    parser.add_argument("--structure", type=float, default=0.5)
    parser.add_argument("--block_structure", type=int, default=15)
    parser.add_argument("--t_structure", type=int, default=0)
    parser.add_argument("--cls_min", type=float, default=21.0)
    parser.add_argument("--cls_type", type=str, default="tanh")
    parser.add_argument(
        "--meandiff_neutral_k",
        type=int,
        default=0,
        help="If > 0 and payload method is subspace_meandiff, strip ΔA along top-k PCA "
        "directions of current layer activations (per forward).",
    )
    parser.add_argument(
        "--dump_baseline_activations",
        type=str,
        default=None,
        help="If set, torch.save pre-steering activations for dump_baseline_prompt_idx, "
        "same dict layout as get_vector activation .pt: {step: {layer_N: {txt|img: (N,T,C)}}} "
        "(N=1) plus string key __dump_meta__. Use --strength 0 --strength_img 0 for baseline.",
    )
    parser.add_argument(
        "--dump_baseline_prompt_idx",
        type=int,
        default=0,
        help="Which prompt index (0-based) to dump when --dump_baseline_activations is set.",
    )
    args = parser.parse_args()

    if args.meandiff_neutral_k < 0:
        raise SystemExit("--meandiff_neutral_k must be >= 0.")

    if args.block_steering != "all":
        args.block_steering = [int(x) for x in args.block_steering.split(",")]
    if args.t_steering != "all":
        args.t_steering = [int(x) for x in args.t_steering.split(",")]

    is_flux = "flux" in args.model_name.lower()
    pipe = (FluxPipeline if is_flux else StableDiffusion3Pipeline).from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if is_flux else torch.float16,
        device_map="balanced",
        use_safetensors=True,
    )

    vector_suffix_candidates = [f"_{args.vector_type}.pt", f"{args.vector_type}.pt"]
    vector_candidates = [
        os.path.join(args.data_dir, name)
        for name in os.listdir(args.data_dir)
        if name.endswith(".pt")
        and "text_" not in name
        and any(name.endswith(suf) for suf in vector_suffix_candidates)
    ]
    if not vector_candidates:
        raise FileNotFoundError(
            f"No vector file in '{args.data_dir}' ending with '{args.vector_type}.pt'"
        )
    if len(vector_candidates) > 1:
        raise RuntimeError(f"Multiple candidates: {vector_candidates}")

    vector = torch.load(vector_candidates[0], map_location="cpu")
    print(f"Loaded: {vector_candidates[0]}")

    vector_txt = None
    if args.steer_txt:
        vector_name = os.path.basename(vector_candidates[0])
        vector_no_ext = vector_name[:-len(".pt")]
        if vector_no_ext.endswith(f"_{args.vector_type}"):
            prefix = vector_no_ext[:-len(f"_{args.vector_type}")]
        elif vector_no_ext.endswith(args.vector_type):
            prefix = vector_no_ext[:-len(args.vector_type)]
        else:
            raise RuntimeError(f"Cannot derive prefix from '{vector_name}'")
        prefix = prefix.rstrip("_")
        expected_txt_names = [
            f"{prefix}_text_diff.pt",
            # f"{prefix}text_}.pt",
        ]
        txt_candidates = [
            os.path.join(args.data_dir, n) for n in expected_txt_names
            if os.path.exists(os.path.join(args.data_dir, n))
        ]
        txt_candidates = [
            os.path.join(args.data_dir, f"base_0.85_20_text_diff.pt",)
        ]
        print(txt_candidates)
        if txt_candidates:
            vector_txt = torch.load(txt_candidates[0], map_location="cpu")
            print(f"Text encoder vector: {txt_candidates[0]}")
        else:
            print(f"WARNING: --steer_txt set but no text file found. Tried: {expected_txt_names}")

    txt_steering = {
        "vector": vector_txt,
        "strength": args.strength_txt,
        "task": args.task,
    }

    try:
        with open(args.prompts_path, "r") as f:
            coco_prompts = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        coco_prompts = ["A high quality photo"]

    if args.task == "nudity":
        dataset = load_dataset("AIML-TUDA/i2p", split="train")
        coco_prompts = [sample["prompt"] for sample in dataset]
        coco_seeds = [int(sample["sd_seed"]) for sample in dataset]
    else:
        with open('/home/jovyan/konovalova/steering/coco_seeds.txt', "r") as f:
            coco_seeds = [line.strip() for line in f if line.strip()][:]
        #coco_seeds = [int(args.seed) for _ in range(len(coco_prompts))]
    named_prompts = []
    for idx, prompt in enumerate(coco_prompts):
        # if args.task == "remove":
        #     prompt = prompt + " " + args.remove_prompt
        
        sanitized = prompt.replace(" ", "_").replace("/", "").replace(",", "")[:50]
        suffix = (
            f"s_{args.strength}_simg_{args.strength_img}_"
            f"ltxt_{args.transport_lambda_txt}_limg_{args.transport_lambda_img}_"
            f"v_{args.vector_type}"
        )

        out_path = os.path.join(args.results_dir, "steered", f"{idx:02d}_{sanitized}_{suffix}.png")
        if os.path.exists(out_path):
            named_prompts.append(f"{idx:02d}_{sanitized}_{suffix}.png")
            continue

        seed = int(coco_seeds[idx])

        print(f"{idx + 1}/{len(coco_prompts)}: {prompt[:]}... seed={seed}")
        hook_state, remove_hooks = apply_attention_steering(pipe, args, vector)
        if args.dump_baseline_activations:
            hook_state["__dump_baseline_path__"] = args.dump_baseline_activations
            hook_state["__dump_baseline_prompt_idx__"] = args.dump_baseline_prompt_idx
            hook_state["__current_prompt_idx__"] = idx
            hook_state["__baseline_acts__"] = {}
        generator = torch.Generator().manual_seed(seed)

        def _on_step_end(_pipe, i, _t, callback_kwargs):
            # Keep hook step in sync with diffusion loop:
            # current forward pass uses step i, next pass should see i+1.
            hook_state.update({"step": 0})
            return callback_kwargs

        images = pipe(
            prompt,
            num_inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
            width=args.width,
            height=args.height,
            generator=generator,
            structure_strength=args.structure,
            callback_on_step_end=_on_step_end,
            callback_on_step_end_tensor_inputs=["latents"],
            txt_steering=txt_steering,
        ).images

        os.makedirs(os.path.join(args.results_dir, "steered"), exist_ok=True)
        try:
            images[0].save(out_path)
        except Exception as exc:
            print(f"Save failed idx {idx}: {exc}")

        if len(images) > 1:
            os.makedirs(os.path.join(args.results_dir, "origin"), exist_ok=True)
            images[1].save(os.path.join(args.results_dir, "origin", f"{idx:02d}_orig.png"))
        if (
            args.dump_baseline_activations
            and idx == args.dump_baseline_prompt_idx
        ):
            if float(args.strength) != 0.0 or float(args.strength_img) != 0.0:
                print(
                    "WARNING: --dump_baseline_activations: strength is not 0; "
                    "saved tensors are still pre-hook steering, but run is not a baseline."
                )
            os.makedirs(
                os.path.dirname(os.path.abspath(args.dump_baseline_activations))
                or ".",
                exist_ok=True,
            )
            raw = hook_state.get("__baseline_acts__", {})
            archive = _pack_activation_archive_like_get_vector(raw)
            # archive["__dump_meta__"] = {
            #     "kind": "baseline_activations_pre_steer",
            #     "prompt_idx": idx,
            #     "prompt": prompt,
            #     "seed": seed,
            #     "strength": args.strength,
            #     "strength_img": args.strength_img,
            #     "transport_lambda_txt": args.transport_lambda_txt,
            #     "transport_lambda_img": args.transport_lambda_img,
            #     "injection_point": args.injection_point,
            #     "width": args.width,
            #     "height": args.height,
            #     "inference_steps": args.inference_steps,
            # }
            torch.save(archive, args.dump_baseline_activations)
            print(f"Saved baseline activations (activation-archive layout): {args.dump_baseline_activations}")
        remove_hooks()
        #assert False
    print(len(named_prompts))

    print('==============================================')
    torch.save(named_prompts, os.path.join("named_prompts_block_13_neutral_k_1.pt"))
        
