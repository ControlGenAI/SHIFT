import argparse
import os
import time
import torch
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV

# --- 1. Expanded Classifier Configuration ---
CLASSIFIER_CONFIGS = {
    'l1':       {'model': 'linearsvc', 'penalty': 'l1', 'dual': False, 'loss': 'squared_hinge'},
    'l2':       {'model': 'linearsvc', 'penalty': 'l2', 'dual': True,  'loss': 'squared_hinge'},
    'logistic': {'model': 'logistic',  'penalty': 'l2', 'solver': 'lbfgs'},
    'none':      {'model': 'linear_no_penalty', 'dual': True,  'loss': 'squared_hinge'},
    'rbf':      {'model': 'svc_rbf',   'C': 1.0, 'gamma': 'scale'}, # Non-linear: No coef_
    'rf':       {'model': 'rf',        'n_estimators': 100}         # Non-linear: No coef_
}

# --- 2. Enhanced Training Logic ---
def prepare_data_slice(act, indices, n_samples):
    X = act.squeeze(1)[:, indices].mean(1)[:n_samples].float()
    norm = X.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return X / norm

def train_classifier(X, y, X_test, y_test, config_key, c_val, seed):
    """
    Trains various classifiers and handles cases where coefficients are unavailable.
    """
    cfg = CLASSIFIER_CONFIGS[config_key].copy()
    model_type = cfg.pop('model')
    
    if model_type == 'linearsvc':
        clf = LinearSVC(**cfg, C=c_val, random_state=seed, max_iter=2000)
    elif model_type == 'logistic':
        clf = LogisticRegression(**cfg, C=c_val, random_state=seed, max_iter=1000)
    elif model_type == 'svc_rbf':
        clf = SVC(kernel='rbf', C=c_val,  probability=True, gamma=cfg['gamma'], random_state=seed)
    elif model_type == 'rf':
        clf = RandomForestClassifier(n_estimators=cfg['n_estimators'], random_state=seed)
    else:
        clf = SVC(kernel='linear', probability=True, random_state=seed) 
   
    if model_type in ['linearsvc']:
        
        calibrated = CalibratedClassifierCV(clf, cv=3)
        calibrated.fit(X, y)
        all_coefs = [est.estimator.coef_ for est in calibrated.calibrated_classifiers_]
        coef = torch.from_numpy(np.mean(all_coefs, axis=0)[0]).float()
        score = calibrated.score(X_test, y_test)
    elif model_type in ['linear_no_penalty', 'logistic']:
        clf.fit(X, y)
        score = clf.score(X_test, y_test)
        coef = torch.from_numpy(clf.coef_[0]).float()
    else:
        clf.fit(X, y)
        score = clf.score(X_test, y_test)
        coef = None 
                
    return clf, coef, score

def _mean_txt_per_block(data, timesteps, blocks, n_samples=None, best_tokens=None):
    means = {}
    print(data.keys(), timesteps)
    for step in range(timesteps):
        means[step] = {}
        for block in range(blocks):
            layer = f'layer_{block}'
            if best_tokens is not None:
                indices = best_tokens[step][block]
                act = data[step][layer].squeeze()[:, indices]
            else:
                act = data[step][layer]['txt'].squeeze()

            if n_samples is not None:
                act = act[:n_samples]

            means[step][layer] = act.float().mean(0)
            del data[step][layer]
        del data[step]
    return means


def _mean_txt_per_block_from_path(path, timesteps, blocks, n_samples=None):
    print(f"Loading {path} ...", flush=True)
    t0 = time.perf_counter()
    data = torch.load(path, map_location='cpu')
    print(f"  loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    t1 = time.perf_counter()
    means = _mean_txt_per_block(data, timesteps, blocks, n_samples=n_samples)
    del data
    print(f"  prompt means in {time.perf_counter() - t1:.1f}s", flush=True)
    return means


def calculate_manual_diff(data_pos, data_neg, timesteps, blocks, best_tokens=None, n_samples=None):
    """Mean diff over prompts: mean(pos) - mean(neg) per (step, block)."""
    pos_means = _mean_txt_per_block(data_pos, timesteps, blocks, n_samples, best_tokens)
    neg_means = _mean_txt_per_block(data_neg, timesteps, blocks, n_samples, best_tokens)

    all_diff = {}
    for step in range(timesteps):
        all_diff[step] = {}
        for block in range(blocks):
            layer = f'layer_{block}'
            all_diff[step][layer] = pos_means[step][layer] - neg_means[step][layer]
    return all_diff


def calculate_manual_diff_from_paths(pos_path, neg_path, timesteps, blocks, n_samples=None):
    """Load pos and neg one at a time to keep peak RAM near one activation dump."""
    pos_means = _mean_txt_per_block_from_path(pos_path, timesteps, blocks, n_samples)
    neg_means = _mean_txt_per_block_from_path(neg_path, timesteps, blocks, n_samples)

    all_diff = {}
    for step in range(timesteps):
        all_diff[step] = {}
        for block in range(blocks):
            layer = f'layer_{block}'
            all_diff[step][layer] = pos_means[step][layer] - neg_means[step][layer]
    return all_diff

# --- 3. Integrated Functional Logic ---

def train_ensemble_svms_best_tokens(data_pos, data_neg, best_tokens, args):
    models, normals = {}, {}
    scores_array = np.zeros((args.n_ensemble, args.timesteps, args.blocks))

    for step in range(args.timesteps):
        models[step], normals[step] = {}, {}
        
        for block in range(args.blocks):
            
            layer = f'layer_{block}'
            indices = best_tokens[step][block]['txt'] if best_tokens else torch.arange(data_pos[0][layer]['txt'].shape[-2])
            if isinstance(indices, torch.Tensor):
                indices = indices.to(dtype=torch.long, device='cpu')
           
            if len(indices) == 0:
                models[step][f'layer_{block}'] = None
                normals[step][f'layer_{block}'] = None
                continue 
            
            X_p = prepare_data_slice(data_pos[step][layer]['txt'], indices, args.n_samples)
            X_n = prepare_data_slice(data_neg[step][layer]['txt'], indices, args.n_samples)
            X, y = np.vstack([X_p.cpu().numpy(), X_n.cpu().numpy()]), np.concatenate([np.ones(len(X_p)), np.zeros(len(X_n))])
            #print(torch.from_numpy(X).norm(dim=-1))
            coefs = []
            models_ensemble = []
            for i in range(args.n_ensemble):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, 
                    test_size=0.4, 
                    random_state=args.random_seed_base + i,
                    stratify=y, 
                    shuffle=True,
                )
            
                m, c, s = train_classifier(X_train, y_train, X_test, y_test, args.classifier, args.c_val, args.random_seed_base + i)
                models_ensemble.append(m)
                if c is not None:
                    coefs.append(c)
                scores_array[i, step, block] = s
            
            if coefs:
                normals[step][layer] = torch.stack(coefs)
            else:
                normals[step][layer] = None 

            models[step][layer] = models_ensemble
            
            print(f"[{args.classifier}] Step {step}, Block {block}. Mean Score: {np.mean(scores_array[:, step, block]):.4f}, {X_train.shape}")
            
    return models, normals, scores_array


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pos_path', required=True)
    parser.add_argument('--neg_path', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--n_samples', type=int, default=20)
    parser.add_argument('--timesteps', type=int, default=4)
    parser.add_argument('--blocks', type=int, default=19)
    parser.add_argument('--method', choices=['svm', 'diff', 'text'], default='svm')
    parser.add_argument('--c_val', type=float, default=0.1)
    parser.add_argument('--n_ensemble', type=int, default=2)
    parser.add_argument('--threshold', type=float, default=0.85)
    parser.add_argument('--classifier', type=str, default='none', choices=['none', 'l1', 'l2', 'logistic', 'rbf', 'rf'])
    parser.add_argument('--random_seed_base', type=int, default=42)
    parser.add_argument('--best_tokens', action='store_true')
    parser.add_argument('--separate_normals', action='store_true')
    parser.add_argument('--save_svm', action='store_true')
    
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    best_tokens, name = None, 'base'
    if args.best_tokens: name += '_best_tokens'
    prefix = os.path.join(args.save_dir, f"{name}_{args.threshold}_{args.n_samples}")

    if args.method == 'diff' and not args.best_tokens:
        all_diff = calculate_manual_diff_from_paths(
            args.pos_path, args.neg_path, args.timesteps, args.blocks, args.n_samples
        )
        out_path = f"{prefix}_diff.pt"
        print(f"Saving {out_path} ...", flush=True)
        torch.save(all_diff, out_path)
        print(f"Saved {out_path}", flush=True)
        return

    data_pos = torch.load(args.pos_path, map_location='cpu')
    data_neg = torch.load(args.neg_path, map_location='cpu')

    if args.method == 'text':
        diff = {k: data_pos[k] - data_neg[k] for k in ['sequence', 'pooled'] if k in data_pos}
        for key in diff:
            print(diff[key].shape)
        torch.save(diff, f"{prefix}_text_diff.pt")
        return

    if args.method == 'diff':
        all_diff = calculate_manual_diff(
            data_pos, data_neg, args.timesteps, args.blocks, best_tokens, args.n_samples
        )
        torch.save(all_diff, f"{prefix}_diff.pt")
        return

    models, normals, scores = train_ensemble_svms_best_tokens(data_pos, data_neg, best_tokens, args)
    torch.save(normals, f'{prefix}_normals.pt')
    torch.save(scores, f'{prefix}_scores.pt')
    torch.save(models, f'{prefix}_svm_models.pt')
    
    print(f"\n✅ Training Complete. Final Mean Accuracy: {np.mean(scores):.4f}")

if __name__ == '__main__':
    main()