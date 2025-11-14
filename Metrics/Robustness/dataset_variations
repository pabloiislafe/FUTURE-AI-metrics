import numpy as np
from sklearn.metrics import (
    accuracy_score as skl_acc,
    precision_score as skl_prec,
    recall_score as skl_rec,
    f1_score as skl_f1,
    roc_auc_score
)

def compute_roc_auc(y_true, y_score, multi_class="ovr", average="macro"):
    return roc_auc_score(
        y_true,
        y_score,
        multi_class=multi_class,
        average=average
    )

def summarize_confusion(mask_true, mask_pred, ignore_mask=None):
    """
    Compute TP/FP/FN/TN for segmentation masks.
    Works only if inputs are multi-dimensional (>=2D).
    """
    ref = np.asarray(mask_true, dtype=bool)
    prd = np.asarray(mask_pred, dtype=bool)

    if ignore_mask is None:
        valid = np.ones_like(ref, dtype=bool)
    else:
        valid = ~np.asarray(ignore_mask, dtype=bool)

    tp = np.sum(( ref &  prd) & valid)
    fp = np.sum((~ref &  prd) & valid)
    fn = np.sum(( ref & ~prd) & valid)
    tn = np.sum((~ref & ~prd) & valid)

    return tp, fp, fn, tn


def compute_accuracy(y_true, y_pred, ignore_mask=None, eps=1e-8):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Classification mode
    if y_true.ndim == 1:
        return skl_acc(y_true, y_pred)

    # Segmentation mode
    tp, fp, fn, tn = summarize_confusion(y_true, y_pred, ignore_mask)
    return (tp + tn) / (tp + fp + fn + tn + eps)

def compute_precision(y_true, y_pred, ignore_mask=None, average="binary", eps=1e-8):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Classification mode
    if y_true.ndim == 1:
        return skl_prec(y_true, y_pred, average=average, zero_division=0)

    # Segmentation mode
    tp, fp, fn, tn = summarize_confusion(y_true, y_pred, ignore_mask)
    return tp / (tp + fp + eps)


def compute_recall(y_true, y_pred, ignore_mask=None, average="binary", eps=1e-8):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Classification mode
    if y_true.ndim == 1:
        return skl_rec(y_true, y_pred, average=average, zero_division=0)

    # Segmentation mode
    tp, fp, fn, tn = summarize_confusion(y_true, y_pred, ignore_mask)
    return tp / (tp + fn + eps)


def compute_f1(y_true, y_pred, ignore_mask=None, average="binary", eps=1e-8):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Classification mode
    if y_true.ndim == 1:
        return skl_f1(y_true, y_pred, average=average, zero_division=0)

    # Segmentation mode → Dice coefficient
    tp, fp, fn, _ = summarize_confusion(y_true, y_pred, ignore_mask)
    return (2 * tp) / (2 * tp + fp + fn + eps)


def compute_iou(y_true, y_pred, ignore_mask=None, eps=1e-8):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # IoU is undefined for classification
    if y_true.ndim == 1:
        raise ValueError("IoU is only defined for segmentation masks.")

    tp, fp, fn, _ = summarize_confusion(y_true, y_pred, ignore_mask)
    return tp / (tp + fp + fn + eps)

def compute_dice(y_true, y_pred, ignore_mask=None, eps=1e-8):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # IoU is undefined for classification
    if y_true.ndim == 1:
        raise ValueError("IoU is only defined for segmentation masks.")

    tp, fp, fn, _ = summarize_confusion(y_true, y_pred, ignore_mask)
    return 2 * tp / (2 * tp + fp + fn)
