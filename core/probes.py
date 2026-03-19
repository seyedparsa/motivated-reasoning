import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy

from xrfm import RFM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def preds_to_proba(preds, eps=1e-3, proba_beta=50):
    if preds.shape[1] == 1:
        activated = F.softplus(preds, beta=proba_beta)
        preds = activated - F.softplus(activated - 1, beta=proba_beta)
    else:
        min_preds = preds.min(dim=1, keepdim=True).values
        max_preds = preds.max(dim=1, keepdim=True).values
        preds = (preds - min_preds) / (max_preds - min_preds + 1e-8)
        preds = torch.clamp(preds, eps, 1-eps)
        preds /= preds.sum(dim=1, keepdim=True)
    return preds


def batch_transpose_multiply(A, B, mb_size=5000):
    n = len(A)
    assert(len(A) == len(B))
    batches = torch.split(torch.arange(n), mb_size)
    sum = 0.
    for b in batches:
        Ab = A[b].cuda()
        Bb = B[b].cuda()
        sum += Ab.T @ Bb

        del Ab, Bb
    return sum


def accuracy_fn(preds, truth):
    assert(len(preds)==len(truth))
    true_shape = truth.shape

    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(truth, torch.Tensor):
        truth = truth.cpu().numpy()

    preds = preds.reshape(true_shape)

    if preds.shape[1] == 1:
        preds = np.where(preds >= 0.5, 1, 0)
        truth = np.where(truth >= 0.5, 1, 0)
    else:
        preds = np.argmax(preds, axis=1)
        truth = np.argmax(truth, axis=1)

    acc = np.sum(preds==truth)/len(preds) * 100
    return acc


def precision_score(preds, labels):
    true_positives = np.sum((preds == 1) & (labels == 1))
    predicted_positives = np.sum(preds == 1)
    return true_positives / (predicted_positives + 1e-8)


def recall_score(preds, labels):
    true_positives = np.sum((preds == 1) & (labels == 1))
    actual_positives = np.sum(labels == 1)
    return true_positives / (actual_positives + 1e-8)


def f1_score(preds, labels):
    precision = precision_score(preds, labels)
    recall = recall_score(preds, labels)
    return 2 * (precision * recall) / (precision + recall + 1e-8)


def compute_prediction_metrics(preds, labels, classification_threshold=0.5):
    if len(labels.shape) == 1:
        labels = labels.reshape(-1, 1)
    num_classes = labels.shape[1]
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    auc = roc_auc_score(labels, preds)
    mse = np.mean((preds-labels)**2)
    if num_classes == 1:
        preds = np.where(preds >= classification_threshold, 1, 0)
        labels = np.where(labels >= classification_threshold, 1, 0)
        acc = accuracy_fn(preds, labels)
        precision = precision_score(preds, labels)
        recall = recall_score(preds, labels)
        f1 = f1_score(preds, labels)
    else:
        preds_classes = np.argmax(preds, axis=1)
        label_classes = np.argmax(labels, axis=1)

        acc = np.sum(preds_classes == label_classes)/ len(preds) * 100

        precision, recall, f1 = 0.0, 0.0, 0.0
        for class_idx in range(num_classes):
            class_preds = (preds_classes == class_idx).astype(np.float32)
            class_labels = (label_classes == class_idx).astype(np.float32)

            precision += precision_score(class_preds, class_labels)
            recall += recall_score(class_preds, class_labels)
            f1 += f1_score(class_preds, class_labels)

        precision /= num_classes
        recall /= num_classes
        f1 /= num_classes

    metrics = {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc, 'mse': mse}
    return metrics


def append_one(X):
    Xb = torch.concat([X, torch.ones_like(X[:,0]).unsqueeze(1)], dim=1)
    new_shape = X.shape[:1] + (X.shape[1]+1,)
    assert(Xb.shape == new_shape)
    return Xb


def train_rfm_probe_on_concept(train_X, train_y, val_X, val_y,
                               hyperparams, search_space=None,
                               tuning_metric='auc'):
    print(f"Training RFM probe on concept with hyperparams: {hyperparams}", flush=True)
    if search_space is None:
        search_space = {
            'regs': [1e-3],
            'bws': [1, 10, 100],
            'center_grads': [True, False]
        }

    best_model = None
    maximize_metric = (tuning_metric in ['f1', 'auc', 'accuracy', 'top_agop_vectors_ols_auc'])
    best_score = float('-inf') if maximize_metric else float('inf')
    combo_idx = 0
    total_combos = len(search_space['regs']) * len(search_space['bws']) * len(search_space['center_grads'])
    for reg in search_space['regs']:
        for bw in search_space['bws']:
            for center_grads in search_space['center_grads']:
                combo_idx += 1
                print(f"  RFM combo {combo_idx}/{total_combos}: reg={reg}, bw={bw}, center_grads={center_grads}", flush=True)
                try:
                    rfm_params = {
                        'model': {
                            'kernel': 'l2_high_dim',
                            'bandwidth': bw,
                            'tuning_metric': tuning_metric,
                        },
                        'fit': {
                            'reg': reg,
                            'iters': hyperparams['rfm_iters'],
                            'center_grads': center_grads,
                            'early_stop_rfm': True,
                            'get_agop_best_model': True,
                            'top_k': hyperparams['n_components']
                        }
                    }
                    model = RFM(**rfm_params['model'], device='cuda')
                    print(f"    Fitting RFM model...", flush=True)
                    model.fit((train_X, train_y),
                              (val_X, val_y),
                              **rfm_params['fit']
                            )
                    print(f"    RFM fit complete", flush=True)

                    if tuning_metric == 'top_agop_vectors_ols_auc':
                        top_k = hyperparams['n_components']
                        targets = val_y

                        _, U = torch.lobpcg(model.agop_best_model, k=top_k)
                        top_eigenvectors = U[:, :top_k]
                        projections = val_X @ top_eigenvectors
                        projections = projections.reshape(-1, top_k)

                        XtX = projections.T @ projections
                        Xty = projections.T @ targets
                        betas = torch.linalg.pinv(XtX) @ Xty
                        preds = torch.sigmoid(projections @ betas).reshape(targets.shape)
                        val_score = roc_auc_score(targets.cpu().numpy(), preds.cpu().numpy())
                    else:
                        pred_proba = model.predict(val_X)
                        val_score = compute_prediction_metrics(pred_proba, val_y)[tuning_metric]

                    if maximize_metric and val_score > best_score or not maximize_metric and val_score < best_score:
                        best_score = val_score
                        best_reg = reg
                        best_model = deepcopy(model)
                        best_bw = bw
                        best_center_grads = center_grads
                except Exception as e:
                    import traceback
                    print(f'Error fitting RFM: {traceback.format_exc()}')
                    continue

    print(f'Best RFM {tuning_metric}: {best_score}, reg: {best_reg}, bw: {best_bw}, center_grads: {best_center_grads}')

    return best_model


def train_linear_probe_on_concept(train_X, train_y, val_X, val_y, use_bias=False, tuning_metric='auc', device='cuda'):
    print(f"Training linear probe on concept with use_bias: {use_bias}, tuning_metric: {tuning_metric}", flush=True)
    reg_search_space = [1e-3, 1e-2, 1e-1, 1, 1e1]

    if use_bias:
        X = append_one(train_X)
        Xval = append_one(val_X)
    else:
        X = train_X
        Xval = val_X

    n, d = X.shape
    num_classes = train_y.shape[1]

    best_beta = None
    best_reg = None
    maximize_metric = (tuning_metric in ['f1', 'auc', 'accuracy'])
    best_score = float('-inf') if maximize_metric else float('inf')
    for reg in reg_search_space:
        try:
            if n>d:
                XtX = batch_transpose_multiply(X, X)
                XtY = batch_transpose_multiply(X, train_y)
                beta = torch.linalg.solve(XtX + reg*torch.eye(X.shape[1], device=X.device), XtY)
            else:
                X = X.to(device)
                train_y = train_y.to(device)
                Xval = Xval.to(device)

                XXt = X@X.T
                alpha = torch.linalg.lstsq(XXt + reg*torch.eye(X.shape[0]).to(device), train_y).solution
                beta = X.T@alpha

            preds = Xval.to(device) @ beta
            preds_proba = preds_to_proba(preds)
            val_score = compute_prediction_metrics(preds_proba, val_y)[tuning_metric]

            if maximize_metric and val_score > best_score or not maximize_metric and val_score < best_score:
                best_score = val_score
                best_reg = reg
                best_beta = deepcopy(beta)

        except Exception as e:
            import traceback
            print(f'Error fitting linear probe: {traceback.format_exc()}')
            continue

    print(f'Linear probe {tuning_metric}: {best_score}, reg: {best_reg}')

    if best_reg is None:
        raise ValueError("Linear probe training failed: no valid regularization found")

    if use_bias:
        line = best_beta[:-1].to(train_X.device)
        if num_classes == 1:
            bias = best_beta[-1].item()
        else:
            bias = best_beta[-1]
    else:
        line = best_beta.to(train_X.device)
        bias = 0

    return line, bias


def train_logistic_probe_on_concept(train_X, train_y, val_X, val_y, use_bias=False, num_classes=1, tuning_metric='auc'):
    print(f"Training logistic probe on concept with use_bias: {use_bias}, num_classes: {num_classes}, tuning_metric: {tuning_metric}", flush=True)
    C_search_space = [1000, 100, 10, 1, 1e-1, 1e-2]

    val_y = val_y.cpu()
    if num_classes == 1:
        train_y_flat = train_y.squeeze(1).cpu()
    else:
        train_y_flat = train_y.argmax(dim=1).cpu()

    best_beta = None
    best_bias = None
    maximize_metric = (tuning_metric in ['f1', 'auc', 'accuracy'])
    best_score = float('-inf') if maximize_metric else float('inf')
    for C in C_search_space:
        model = LogisticRegression(fit_intercept=use_bias, max_iter=2000, C=C)
        model.fit(train_X.cpu(), train_y_flat.cpu())

        val_probs = torch.tensor(model.predict_proba(val_X.cpu()))
        if num_classes == 1:
            val_probs = val_probs[:,1].reshape(val_y.shape)
        val_score = compute_prediction_metrics(val_probs, val_y)[tuning_metric]

        if maximize_metric and val_score > best_score or not maximize_metric and val_score < best_score:
            best_score = val_score
            best_beta = torch.from_numpy(model.coef_).T
            if use_bias:
                best_bias = torch.from_numpy(model.intercept_)
            best_C = C

    print(f'Logistic probe {tuning_metric}: {best_score}, C: {best_C}')
    best_beta = best_beta.to(train_X.device).float()
    best_bias = best_bias.to(train_X.device).float()
    if use_bias:
        line = best_beta
        if num_classes == 1:
            bias = best_bias.item()
        else:
            bias = best_bias
    else:
        line = best_beta
        bias = 0

    return line, bias
