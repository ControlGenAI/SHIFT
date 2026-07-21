import argparse
import os
from typing import Iterable

import numpy as np
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC


CLASSIFIER_CONFIGS = {
    "l1": {"model": "linearsvc", "penalty": "l1", "dual": False, "loss": "squared_hinge"},
    "l2": {"model": "linearsvc", "penalty": "l2", "dual": True, "loss": "squared_hinge"},
    "logistic": {"model": "logistic", "penalty": "l2", "solver": "lbfgs"},
    "none": {"model": "linear_no_penalty", "dual": True, "loss": "squared_hinge"},
    "rbf": {"model": "svc_rbf", "C": 1.0, "gamma": "scale"},
    "rf": {"model": "rf", "n_estimators": 100},
}


def to_sample_tensor(act: torch.Tensor, n_samples: int = None) -> torch.Tensor:
    act = act.float()
    while act.dim() > 3 and act.shape[1] == 1:
        act = act.squeeze(1)
    if act.dim() == 2:
        act = act.unsqueeze(0)
    elif act.dim() > 3:
        act = act.reshape(-1, act.shape[-2], act.shape[-1])
    if n_samples is not None:
        act = act[:n_samples]
    if act.dim() != 3:
        raise ValueError(f"Expected (n_samples, n_tokens, d), got shape {tuple(act.shape)}")
    return act


def prepare_token_tensor(act: torch.Tensor, indices, n_samples: int = None) -> torch.Tensor:
    x = to_sample_tensor(act, n_samples=n_samples)
    if indices is not None:
        if isinstance(indices, torch.Tensor):
            indices = indices.to(dtype=torch.long, device="cpu")
        x = x[:, indices]
    return x


def prepare_data_slice(act: torch.Tensor, indices, n_samples: int = None) -> torch.Tensor:
    x = prepare_token_tensor(act, indices, n_samples)
    x = x.mean(dim=1)
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-6)


def train_classifier(X, y, X_test, y_test, config_key, c_val, seed):
    cfg = CLASSIFIER_CONFIGS[config_key].copy()
    model_type = cfg.pop("model")

    if model_type == "linearsvc":
        clf = LinearSVC(**cfg, C=c_val, random_state=seed, max_iter=2000)
    elif model_type == "logistic":
        clf = LogisticRegression(**cfg, C=c_val, random_state=seed, max_iter=1000)
    elif model_type == "svc_rbf":
        clf = SVC(kernel="rbf", C=c_val, probability=True, gamma=cfg["gamma"], random_state=seed)
    elif model_type == "rf":
        clf = RandomForestClassifier(n_estimators=cfg["n_estimators"], random_state=seed)
    else:
        clf = SVC(kernel="linear", probability=True, random_state=seed)

    if model_type == "linearsvc":
        calibrated = CalibratedClassifierCV(clf, cv=3)
        calibrated.fit(X, y)
        all_coefs = [est.estimator.coef_ for est in calibrated.calibrated_classifiers_]
        coef = torch.from_numpy(np.mean(all_coefs, axis=0)[0]).float()
        score = calibrated.score(X_test, y_test)
        return calibrated, coef, score

    clf.fit(X, y)
    score = clf.score(X_test, y_test)
    coef = torch.from_numpy(clf.coef_[0]).float() if hasattr(clf, "coef_") else None
    return clf, coef, score


def iter_branches(pos_layer, neg_layer, branches: Iterable[str]):
    if isinstance(pos_layer, dict):
        for branch in branches:
            if branch in pos_layer and branch in neg_layer:
                yield branch, pos_layer[branch], neg_layer[branch]
    else:
        yield "txt", pos_layer, neg_layer


def get_best_indices(best_tokens, step: int, block: int, branch: str):
    if not best_tokens:
        return None
    val = best_tokens[step][block]
    if isinstance(val, dict):
        return val.get(branch)
    return val


def calculate_manual_diff(data_pos, data_neg, args, best_tokens=None):
    out = {}
    branches = ("img", "txt") if args.token_stream == "both" else (args.token_stream,)
    for step in range(args.timesteps):
        out[step] = {}
        for block in range(args.blocks):
            layer = f"layer_{block}"
            if step not in data_pos or layer not in data_pos[step]:
                continue
            if step not in data_neg or layer not in data_neg[step]:
                continue
            layer_out = {}
            for branch, pos_act, neg_act in iter_branches(data_pos[step][layer], data_neg[step][layer], branches):
                indices = get_best_indices(best_tokens, step, block, branch)
                pos = prepare_token_tensor(pos_act, indices, args.n_samples)
                neg = prepare_token_tensor(neg_act, indices, args.n_samples)
                layer_out[branch] = (pos - neg).mean(dim=0)
            if layer_out:
                out[step][layer] = layer_out
    return out


def train_ensemble_svms(data_pos, data_neg, best_tokens, args):
    models, normals, scores = {}, {}, {}
    branches = ("img", "txt") if args.token_stream == "both" else (args.token_stream,)

    for step in range(args.timesteps):
        models[step], normals[step], scores[step] = {}, {}, {}
        for block in range(args.blocks):
            layer = f"layer_{block}"
            if step not in data_pos or layer not in data_pos[step]:
                continue
            if step not in data_neg or layer not in data_neg[step]:
                continue

            models[step][layer], normals[step][layer], scores[step][layer] = {}, {}, {}
            for branch, pos_act, neg_act in iter_branches(data_pos[step][layer], data_neg[step][layer], branches):
                indices = get_best_indices(best_tokens, step, block, branch)
                X_p = prepare_data_slice(pos_act, indices, args.n_samples)
                X_n = prepare_data_slice(neg_act, indices, args.n_samples)
                X = np.vstack([X_p.cpu().numpy(), X_n.cpu().numpy()])
                y = np.concatenate([np.ones(len(X_p)), np.zeros(len(X_n))])

                coefs, ensemble, branch_scores = [], [], []
                for i in range(args.n_ensemble):
                    X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        y,
                        test_size=0.4,
                        random_state=args.random_seed_base + i,
                        stratify=y,
                        shuffle=True,
                    )
                    model, coef, score = train_classifier(
                        X_train, y_train, X_test, y_test, args.classifier, args.c_val, args.random_seed_base + i
                    )
                    ensemble.append(model)
                    branch_scores.append(score)
                    if coef is not None:
                        coefs.append(coef)

                models[step][layer][branch] = ensemble
                normals[step][layer][branch] = torch.stack(coefs) if coefs else None
                scores[step][layer][branch] = branch_scores
                print(f"[{args.classifier}] Step {step}, Block {block}, {branch}. Mean Score: {np.mean(branch_scores):.4f}")

    return models, normals, scores


def main():
    parser = argparse.ArgumentParser(description="Calculate SD3.5 dual-branch steering vectors.")
    parser.add_argument("--pos_path", required=True)
    parser.add_argument("--neg_path", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--timesteps", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=24)
    parser.add_argument("--token_stream", choices=["img", "txt", "both"], default="both")
    parser.add_argument("--method", choices=["svm", "diff", "text"], default="svm")
    parser.add_argument("--c_val", type=float, default=0.1)
    parser.add_argument("--n_ensemble", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--classifier", type=str, default="none", choices=["none", "l1", "l2", "logistic", "rbf", "rf"])
    parser.add_argument("--random_seed_base", type=int, default=42)
    parser.add_argument("--best_tokens_path", type=str, default=None)
    parser.add_argument("--save_svm", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    data_pos = torch.load(args.pos_path, map_location="cpu")
    data_neg = torch.load(args.neg_path, map_location="cpu")
    best_tokens = torch.load(args.best_tokens_path, map_location="cpu") if args.best_tokens_path else None

    name = f"sd35_base_{args.token_stream}"
    prefix = os.path.join(args.save_dir, f"{name}_{args.threshold}_{args.n_samples}")

    if args.method == "text":
        diff = {k: data_pos[k] - data_neg[k] for k in ["sequence", "pooled"] if k in data_pos}
        torch.save(diff, f"{prefix}_text_diff.pt")
        return

    if args.method == "diff":
        torch.save(calculate_manual_diff(data_pos, data_neg, args, best_tokens), f"{prefix}_diff.pt")
        return

    models, normals, scores = train_ensemble_svms(data_pos, data_neg, best_tokens, args)
    torch.save(normals, f"{prefix}_normals.pt")
    torch.save(scores, f"{prefix}_scores.pt")
    if args.save_svm:
        torch.save(models, f"{prefix}_svm_models.pt")
    print("Training complete.")


if __name__ == "__main__":
    main()
