import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from metrics.robustness.dataset_variations import summarize_confusion

def compute_ssim(img_ref: np.ndarray, 
                 img_pred: np.ndarray, 
                 data_range=None):
    """
    Structural Similarity Index (SSIM).

    img_ref, img_pred: images with same shape (2D or 3D).
    data_range: optional, max-min intensity.

    Returns: float in [-1, 1], usually [0,1].
    """
    img_ref = img_ref.astype(np.float32)
    img_pred = img_pred.astype(np.float32)

    if data_range is None:
        data_range = img_ref.max() - img_ref.min()

    # skimage handles 2D directly, for 3D we compute slice-wise mean SSIM
    if img_ref.ndim == 2:
        return ssim(img_ref, img_pred, data_range=data_range)
    elif img_ref.ndim == 3:
        values = [
            ssim(img_ref[..., i], img_pred[..., i], data_range=data_range)
            for i in range(img_ref.shape[-1])
        ]
        return np.mean(values)
    else:
        raise ValueError("SSIM only supports 2D or 3D images.")

def compute_psnr(img_ref: np.ndarray, img_pred: np.ndarray, data_range=None):
    """
    PSNR using skimage.metrics.peak_signal_noise_ratio.
    
    img_ref, img_pred: images of same shape
    data_range: max-min intensity (optional)
    """
    img_ref = img_ref.astype(np.float32)
    img_pred = img_pred.astype(np.float32)

    if data_range is None:
        data_range = img_ref.max() - img_ref.min()

    return psnr(img_ref, img_pred, data_range=data_range)

def compute_specificity(mask_ref, mask_pred, ignore_mask=None, eps=1e-8):
    """
    Specificity = TN / (TN + FP)
    """
    tp, fp, fn, tn = summarize_confusion(mask_ref, mask_pred, ignore_mask)
    return tn / (tn + fp + eps)

def compute_fpr(mask_ref, mask_pred, ignore_mask=None, eps=1e-8):
    """
    False Positive Rate (FPR) = FP / (FP + TN)
    """
    tp, fp, fn, tn = summarize_confusion(mask_ref, mask_pred, ignore_mask)
    return fp / (fp + tn + eps)
