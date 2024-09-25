import typing as tp 

import gin 
import sklearn.metrics
import torch
import torch.nn as nn 
from tqdm import tqdm
import torcheval.metrics 
import sklearn

@torch.no_grad()
def compute_accuracy(preds: torch.Tensor, 
                     labels: torch.Tensor)->float:
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    preds = preds.argmax(-1)
    num_samples = labels.shape[0]
    mispreds = torch.where(preds!=labels, 1, 0).sum()
    return 1-(mispreds/num_samples)

@gin.register(module='metrics')
@torch.no_grad()
def compute_weighted_accuracy(preds: torch.Tensor, 
                              labels: torch.Tensor)->float: 
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    # preds = preds.argmax(-1)
    weighted_acc = torcheval.metrics.MulticlassAccuracy(num_classes=preds.shape[-1], average='macro')
    weighted_acc.update(preds, labels)
    return weighted_acc.compute()

@gin.register(module='metrics', allowlist=['do_sigmoid'])
@torch.no_grad()
def compute_average_precision(preds: torch.Tensor, 
                              labels: torch.Tensor, 
                              do_sigmoid: bool = True)->float:
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    if do_sigmoid:
        preds = torch.sigmoid(preds)
    avg_score = sklearn.metrics.average_precision_score(labels, preds, average=None)
    return avg_score.mean()

@gin.register(module='metrics')
@torch.no_grad()
def compute_auroc(preds: torch.Tensor, 
                  labels: torch.Tensor)->float: 
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    binary_auroc = torcheval.metrics.BinaryAUROC(device='cpu', num_tasks=preds.shape[-1])
    binary_auroc.update(preds.permute(1, 0), labels.permute(1, 0))
    return binary_auroc.compute().mean().item()

@gin.register(module='metrics', allowlist=['tokenizer'])
@torch.no_grad()
def compute_wer(preds: torch.Tensor, 
                labels: torch.Tensor, 
                tokenizer: tp.Callable[[torch.Tensor], str] = None)->float:
    wer = torcheval.metrics.WordErrorRate()
    preds = [pred.argmax(1) for pred in preds]
    preds = [p.cpu().tolist() for pred in preds for p in pred]
    labels = [l.cpu().tolist() for label in labels for l in label]
    for pred, label in zip(preds, labels):
        txt_pred = tokenizer.decode_single(pred)
        txt_true = tokenizer.decode_single(label)
        wer.update([txt_pred], [txt_true])
    return wer.compute().item()

def compute_metrics(model: nn.Module, 
                    device: str, 
                    loader: tp.Iterable[tp.Tuple[torch.Tensor, tp.List[str], torch.Tensor]], 
                    metrics: tp.Dict[str, tp.Callable[[tp.Any], tp.Any]],
                    pbar: bool = True) -> tp.Dict[str, float]:
    model.eval()
    model = model.to(device)
    preds = []
    labels = []
    if pbar:
        loader = tqdm(loader, desc=f'Evaluating model on metrics {tuple(metrics.keys())}', leave=False)
    for x, label, _ in loader:
        labels.append(label)
        x = x.to(device)
        pred = model(x).detach().cpu()
        preds.append(pred)
    results = {}
    for name, metric_fn in metrics.items():
        results[name] = metric_fn(preds, labels)
    return results