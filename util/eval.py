from PIL import Image
import numpy as np
import torch

def _binarize(arr):
    arr = np.asarray(arr)
    if arr.dtype == object:
        arr = arr.astype(np.float32)
    arr = np.squeeze(arr)
    return np.array(Image.fromarray((arr * 255).astype(np.uint8)).convert('L').convert('1'))


def dice_coefficient(y_true, y_pred):
    """Dice coefficient: 2*|A&B| / (|A|+|B|)."""
    y_true_f = _binarize(y_true).flatten()
    y_pred_f = _binarize(y_pred).flatten()
    smooth = 1  # Laplace smoothing to avoid division by zero
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def iou_coefficient(y_true, y_pred):
    """IoU = intersection / union."""
    y_true_f = _binarize(y_true).flatten()
    y_pred_f = _binarize(y_pred).flatten()

    smooth = 1  # Laplace smoothing to avoid division by zero
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection

    return (intersection + smooth) / (union + smooth)

def getdefect(rawimage, prior, k):
    if isinstance(rawimage, torch.Tensor):
        rawimage = rawimage.detach().cpu().numpy()
    if isinstance(prior, torch.Tensor):
        prior = prior.detach().cpu().numpy()
    rawimage = np.asarray(rawimage, dtype=np.float32)
    prior = np.asarray(prior, dtype=np.float32)
    image = np.absolute(rawimage - prior)

    mean_values = image.mean(axis=-1)  
    binary_image = (mean_values > k).astype(np.uint8)  
    
    output_image = np.stack([binary_image] * 3, axis=-1)  
    return output_image
