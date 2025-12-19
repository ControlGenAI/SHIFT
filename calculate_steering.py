import argparse
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from tqdm import tqdm
import os
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


def train_svms_best_tokens(data_pos, data_neg, best_tokens, timesteps=20, blocks=56, n_samples=25, save_dir=None, model_type='flux'):
    """
    Train SVMs using only the best tokens (indices_above_threshold_per_step) for each step and block.
    data_pos, data_neg: dicts of [step][f'layer_{block}'] -> tensor
    best_tokens: dict of [step][block] -> list of indices
    """
    models = {}
    normals = {}
    scores = np.zeros((timesteps, blocks))

    for step in range(timesteps):
        models[step] = {}
        normals[step] = {}
        for block in range(blocks)[:]:
            if best_tokens is not None:
                indices = best_tokens[step][block]
            else:
                if model_type == 'flux':
                    indices = torch.arange(512)
                elif model_type == 'sd3':
                    indices = torch.arange(333)
                else:
                    raise ValueError("no such type of model is avaliable now")

            if len(indices) != 0:
                # --- 2. Prepare Data (Extract, Average, Normalize) ---
                # Positive Samples
                if len(data_pos[step][f'layer_{block}'].shape) == 3:
                    # [Batch, Tokens, Features] -> Select features, average over tokens
                    X_pos = data_pos[step][f'layer_{block}'][:, indices].mean(1)[:n_samples].float()
                else:
                    # [Batch, 1, Tokens, Features] -> Select features, average over tokens
                    # this is for flux single blocks
                    X_pos = data_pos[step][f'layer_{block}'][:, 0, indices].mean(1)[:n_samples].float()
                X_pos = X_pos / X_pos.norm(dim=-1, keepdim=True)

                # Negative Samples
                if len(data_neg[step][f'layer_{block}'].shape) == 3:
                    X_neg = data_neg[step][f'layer_{block}'][:, indices].mean(1)[:n_samples].float()
                else:
                    # this is for flux single blocks
                    X_neg = data_neg[step][f'layer_{block}'][:, 0, indices].mean(1)[:n_samples].float()
                
                
                X_neg = X_neg / X_neg.norm(dim=-1, keepdim=True)

                # --- 3. Train SVM ---
                y_pos = np.ones(X_pos.shape[0])
                y_neg = np.zeros(X_neg.shape[0])
                
                # Convert to numpy for sklearn
                X = np.vstack([X_pos.cpu().numpy(), X_neg.cpu().numpy()])
                y = np.concatenate([y_pos, y_neg])
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True,
                )
                
                # Train SVM
                # clf = SVC(kernel='linear', probability=True)
                # clf.fit(X_train, y_train)

                l1_svc = LinearSVC(
                    penalty='l1',
                    loss='squared_hinge',
                    dual=False,
                    C=0.1,  # Adjust this: smaller C means more zero coefficients
                    random_state=42
                )

                # 2. Instantiate the Calibrated Classifier
                #    - cv=5: Performs 5-fold cross-validation for calibration (like probability=True in SVC).
                #    - method='sigmoid': The most common method for probability calibration.
                clf = CalibratedClassifierCV(
                    estimator=l1_svc,
                    method='sigmoid',
                    #cv=5
                )
                
                # Score
                score = clf.score(X_test, y_test)
                scores[step, block] = score
                # Store normal vector (coefficients)
                coef_tensor = torch.from_numpy(clf.coef_[0]).float()
                if torch.isnan(coef_tensor).any():
                    print(f"NaN detected in normals at step {step}, block {block}")
                    normals[step][f'layer_{block}'] = None
                    models[step][f'layer_{block}'] = None
                    continue
                
                normals[step][f'layer_{block}'] = coef_tensor
                models[step][f'layer_{block}'] = clf
                
                print(step, block, score, X_train.shape, X_test.shape)
            else:
                # Handle case where indices is empty
                models[step][f'layer_{block}'] = None
                normals[step][f'layer_{block}'] = None
    
    return models, normals, scores


# --------------------------------------------------------------------------------------------------


def train_ensemble_svms_best_tokens(data_pos, data_neg, best_tokens, timesteps=20, blocks=56, n_samples=25, n_ensemble=2, test_size=0.2, random_seed_base=42, save_dir=None, model_type='flux'):
    """
    Trains an ensemble of SVMs using only the best tokens for each step and block.
    
    The function returns the ensemble of normals as a tensor of shape 
    (n_ensemble, feature_dimension) for each block/timestep.

    Args:
        data_pos, data_neg: dicts of [step][f'layer_{block}'] -> tensor (activations)
        best_tokens: dict of [step][block] -> list of indices (tokens to use)
        timesteps (int): Number of time steps to iterate over.
        blocks (int): Number of blocks (layers) to iterate over.
        n_samples (int): Number of positive/negative samples to use.
        n_ensemble (int): The number of SVM models to train for the ensemble.
        test_size (float): Proportion of the data to include in the test split.
        random_seed_base (int): Base seed for reproducible data splitting across ensembles.

    Returns:
        models (dict): Nested dictionary of [step][block] -> list of SVC models.
        normals (dict): Nested dictionary of [step][block] -> torch.Tensor of shape (n_ensemble, feature_dim).
        scores (np.array): Test accuracies for the ensemble, shape (n_ensemble, timesteps, blocks).
    """
    models = {}
    normals = {}
    # scores will store the accuracy of each ensemble member for each step/block
    scores = np.zeros((n_ensemble, timesteps, blocks)) 

    for step in range(timesteps):
        models[step] = {}
        normals[step] = {}
        
        for block in range(blocks):
            # print( data_pos[step].keys())
            # assert False
            
            # --- 1. Get Indices ---
            if best_tokens is not None:
                indices = best_tokens[step][block] 
            else:
                # Placeholder logic (adjust based on your feature/token count)
                if model_type == 'flux':
                    indices = torch.arange(512)
                elif model_type == 'sd3':
                    indices = torch.arange(333)
                else:
                    raise ValueError("No such type of model")
                        
            #print(indices)
            if len(indices) == 0:
                models[step][f'layer_{block}'] = None
                normals[step][f'layer_{block}'] = None
                continue 

            
            # --- 2. Prepare Data (Extract, Average, Normalize) ---
            
            def prepare_data(data):
                # Handles 3D ([Batch, Tokens, Features]) or 4D ([Batch, 1, Tokens, Features]) activation shapes
                
                # --- CORRECTION: USE INDICES FOR FEATURE SELECTION ---
                if len(data[step][f'layer_{block}'].shape) == 3:
                    # [Batch, Tokens, Features] -> Select features using indices, average over token dim (dim=1)
                    print(step, block, data[step][f'layer_{block}'][:, indices].shape, data[step][f'layer_{block}'].shape, data[step][f'layer_{block}'][:, indices].mean(1).shape)
                    X = data[step][f'layer_{block}'][:, indices].mean(1)[:n_samples].float()
                else:
                    # [Batch, 1, Tokens, Features] -> Select features using indices, average over token dim (dim=2)
                    print(step, block, data[step][f'layer_{block}'].shape, data[step][f'layer_{block}'].shape, data[step][f'layer_{block}'][:, 0, indices].mean(1).shape)
                    X = data[step][f'layer_{block}'][:, 0, indices].mean(1)[:n_samples].float()
                
                # Normalization (Crucial for consistent steering vectors)
                norm = X.norm(dim=-1, keepdim=True)
                norm[norm == 0] = 1e-6 # Avoid division by zero
                X = X / norm
                return X
                
            X_pos = prepare_data(data_pos)
            X_neg = prepare_data(data_neg)
            
            y_pos = np.ones(X_pos.shape[0])
            y_neg = np.zeros(X_neg.shape[0])
            
            # Combine all samples for splitting (Convert to numpy once)
            X_all = np.vstack([X_pos.cpu().numpy(), X_neg.cpu().numpy()])
            y_all = np.concatenate([y_pos, y_neg])
            
            
            # --- 3. Ensemble Training Loop ---
            
            ensemble_models = []
            ensemble_coefs = []
            ensemble_scores = []
            
            for i in range(n_ensemble):
                # Split data differently for each ensemble member using a unique random_state
                X_train, X_test, y_train, y_test = train_test_split(
                    X_all, y_all, 
                    test_size=test_size, 
                    random_state=random_seed_base + i, # Unique seed for different splits
                    stratify=y_all, 
                    shuffle=True,
                )
                print(X_train.shape, )
                
                # Train model
                # Use a unique random state for the model itself (SVC is deterministic for linear kernel)
                clf = SVC(kernel='linear', probability=True, random_state=random_seed_base + i) 
                add_noise = False

                if add_noise:
                    assert False
                    noise = np.random.normal(loc=0.0, scale=X_train.mean(0).std(), size=X_train.shape)
    
                    # Add noise to the features
                    X_train = X_train + noise * 2

                # l1_svc = LinearSVC(
                #     penalty='l2',
                #     loss='squared_hinge',
                #     dual=False,
                #     C=100000,  # Adjust this: smaller C means more zero coefficients
                #     random_state=42,
                #     max_iter=100000
                # )

                # 2. Instantiate the Calibrated Classifier
                #    - cv=5: Performs 5-fold cross-validation for calibration (like probability=True in SVC).
                #    - method='sigmoid': The most common method for probability calibration.
                # clf = CalibratedClassifierCV(
                #     estimator=l1_svc,
                #     method='sigmoid',
                #     #cv=1
                # )
                clf.fit(X_train, y_train)
                #assert False
                
                # Score and store
                score = clf.score(X_test, y_test)
                ensemble_scores.append(score)
                ensemble_models.append(clf)
                
                # Store coefficient (normal)
                coef_tensor = torch.from_numpy(clf.coef_[0]).float()
                #coef_tensor = torch.from_numpy(clf.calibrated_classifiers_[0].estimator.coef_).float()
                ensemble_coefs.append(coef_tensor)
                print(np.count_nonzero(coef_tensor))
            
            
            # --- 4. Final Aggregation and Storage ---
            
            # Check for potential issues before stacking
            if any(torch.isnan(c).any() for c in ensemble_coefs):
                 print(f"NaN detected in normals in ensemble at step {step}, block {block}. Skipping ensemble.")
                 models[step][f'layer_{block}'] = None
                 normals[step][f'layer_{block}'] = None
                 continue
                 
            # Stack the coefficient tensors to create the final normal tensor: (n_ensemble, feature_dim)
            normal_tensor_ensemble = torch.stack(ensemble_coefs)
            print(step, block)
            
            # Store results
            normals[step][f'layer_{block}'] = normal_tensor_ensemble # Tensor of shape (N, D)
            models[step][f'layer_{block}'] = ensemble_models       # List of N models
            
            mean_score = np.mean(ensemble_scores)
            scores[:, step, block] = ensemble_scores # Store all scores
            
            print(f"Ensemble Training: Step {step}, Block {block}. Mean Score: {mean_score:.4f} (N={n_ensemble}). Normal Shape: {normal_tensor_ensemble.shape}")
        
    return models, normals, scores


# --------------------------------------------------------------------------------------------------


def train_svms_per_token(data_pos, data_neg, timesteps=20, blocks=24, n_samples=50, threshold=0.5, save_dir=None):
    """
    Train SVMs for each feature (token) individually for each step and block.
    data_pos, data_neg: dicts of [step][f'layer_{block}'] -> tensor
    
    Note: The 'feature' in this context is the embedding dimension, not the token. 
    The original logic iterates over the *second* dimension of the data tensor (index 1), which
    corresponds to the sequence/token length dimension, not the feature/embedding dimension (index 2).
    It then selects the last part of this dimension ([77:]).
    The terminology "per_token" seems to mean "per position/sequence index".
    
    Returns: models, normals, scores_correct, indices_above_threshold_per_step
    """
    assert False

    models = {}
    normals = {}
    scores = []
    for step in range(timesteps):
        models[step] = {}
        normals[step] = {} 
        for block in range(blocks):
            print( data_pos[step].keys())
            #assert False
            X_pos_all = data_pos[step][f'layer_{block}'][:n_samples].squeeze()
            X_pos_all = X_pos_all / X_pos_all.norm(dim=-1, keepdim=True)
            X_neg_all = data_neg[step][f'layer_{block}'][:n_samples].squeeze()
            X_neg_all = X_neg_all / X_neg_all.norm(dim=-1, keepdim=True)
            scores_all = []
            models[step][f'layer_{block}'] = []
            normals[step][f'layer_{block}'] = []
            
            for i in tqdm(range(X_pos_all.shape[1]), desc=f"Step {step} Block {block}"):
                
                # Select the i-th token/position across all samples for all features
                # X_pos_all[:, i] has shape [n_samples, Features]
                X_pos = X_pos_all[:, i].cpu().numpy()
                X_neg = X_neg_all[:, i].cpu().numpy()
            
                y_pos = np.ones(X_pos.shape[0])
                y_neg = np.zeros(X_neg.shape[0])
                
                X = np.vstack([X_pos, X_neg])
                y = np.concatenate([y_pos, y_neg])
                
                # Check if enough samples remain after slicing
                if X.shape[0] < 2:
                    continue

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True,
                )
                clf = SVC(kernel='linear', probability=True)
                clf.fit(X_train, y_train)
                coef_tensor = torch.from_numpy(clf.coef_[0]).float()
                
                if torch.isnan(coef_tensor).any():
                    print(f"NaN detected in normals at step {step}, block {block}, token {i}")
                    continue
                normals[step][f'layer_{block}'].append(coef_tensor)
                models[step][f'layer_{block}'].append(clf)
                score = clf.score(X_test, y_test)
                scores_all.append(score)
            scores.append(scores_all)
            print(step, block, np.array(scores_all).mean() if scores_all else 0.0, X_train.shape if 'X_train' in locals() else (0,0), X_test.shape if 'X_test' in locals() else (0,0))
            
    # Reshape scores from flat list to [timesteps, blocks, tokens]
    scores_correct = []
    print(blocks, timesteps)
    for i in range(timesteps):
        scores_correct.append(scores[i*blocks:(i+1)*blocks])
    scores_correct = np.array(scores_correct)
    print(scores_correct.shape)
             
  # set your desired threshold value
    mean_scores = scores_correct.mean(0)  
    # mean_scores shape: (24, 333)

    indices_above_threshold_per_step = []
    for step in range(scores_correct.shape[0]):
        mean_scores = scores_correct[step]
        indices_above_threshold_per_block = []
        for block in range(mean_scores.shape[0]):
            # Use the 0.75 quantile as the threshold instead of a predefined value
            quantile_threshold = min(np.quantile(mean_scores[block], 0.5), threshold)
            print(quantile_threshold)
            indices = np.where(mean_scores[block] > quantile_threshold)[0]
            indices_above_threshold_per_block.append(indices)
        indices_above_threshold_per_step.append(indices_above_threshold_per_block)

    return models, normals, scores_correct, indices_above_threshold_per_step


# --------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description='Train SVMs on attention data')
    parser.add_argument('--pos_path', type=str, required=True, help='Path to positive data .pt file')
    parser.add_argument('--neg_path', type=str, required=True, help='Path to negative data .pt file')
    parser.add_argument('--timesteps', type=int, default=20, help='Number of timesteps')
    parser.add_argument('--gs', type=float, default=4.5, help='Guidance scale (unused)')
    parser.add_argument('--threshold', type=float, default=0.85, help='Threshold for best token selection')
    parser.add_argument('--blocks', type=int, default=24, help='Number of blocks/layers')
    parser.add_argument('--n_samples', type=int, default=25, help='Number of samples per class')
    parser.add_argument('--task', type=int, default=50, help='Task id (unused)')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save models and scores')
    parser.add_argument('--best_tokens', action='store_true', help='Whether to compute best tokens using per-token SVMs')
    parser.add_argument('--separate_normals', action='store_true', help='Whether to save normals and scores for best tokens separately')
    parser.add_argument('--save_svm', action='store_true', help='Whether to save SVMs and normals for best tokens')
    args = parser.parse_args()


    if not os.path.exists(args.pos_path):
        raise FileNotFoundError(f"Positive data file not found: {args.pos_path}")
    if not os.path.exists(args.neg_path):
        raise FileNotFoundError(f"Negative data file not found: {args.neg_path}")
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    try:
        data_pos = torch.load(args.pos_path)
        data_neg = torch.load(args.neg_path)
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")

    best_tokens = None
    normals = None
    scores = None

    name = 'base'

    if args.best_tokens:
        # The output of train_svms_per_token is (models, normals, scores_correct, indices_above_threshold_per_step)
        _, normals_per_token, scores_per_token, best_tokens = train_svms_per_token(
            data_pos, data_neg, timesteps=args.timesteps, blocks=args.blocks, n_samples=args.n_samples, threshold=args.threshold, save_dir=args.save_dir)
        name += '_best_tokens'

    
    if args.separate_normals:
        print('Saving separate normals and scores (from per-token SVMs)...')
        if normals_per_token is not None:
             torch.save(normals_per_token, f'{args.save_dir}/{name}_{args.threshold}_{args.n_samples}_normals_separate.pt')
        if scores_per_token is not None:
             np.save(f'{args.save_dir}/{name}_{args.threshold}_{args.n_samples}_scores_separate.npy', scores_per_token)
        if best_tokens is not None:
             torch.save(best_tokens, f'{args.save_dir}/{name}_{args.threshold}_{args.n_samples}_best_tokens.pt') # Renamed to be clearer

    # Train the ensemble SVMs (optionally with best_tokens from above)
    if not args.separate_normals or args.save_svm:
        print("Training Ensemble SVMs and saving results...")
        models, normals, scores = train_ensemble_svms_best_tokens(
            data_pos, data_neg, best_tokens, timesteps=args.timesteps, blocks=args.blocks, n_samples=args.n_samples, save_dir=args.save_dir)

        # Save results from ensemble SVMs
        torch.save(models, f'{args.save_dir}/{name}_{args.threshold}_{args.n_samples}_svm_models.pt')
        torch.save(normals, f'{args.save_dir}/{name}_{args.threshold}_{args.n_samples}_normals.pt')
        torch.save(scores, f'{args.save_dir}/{name}_{args.threshold}_{args.n_samples}_scores.pt')
    
    print(f"Final Configuration: Samples per Class: {args.n_samples}, Threshold: {args.threshold}")

if __name__ == '__main__':
    main()