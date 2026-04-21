import os
# define the utils
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def post_process(img):
    img=img.clone()
    img = denorm(img[0])
    img = img.permute(1, 2, 0)
    return (img.detach().cpu().numpy().astype(np.float32))

def denorm(img):
    for i in range(img.shape[0]):
        img[i] = img[i] * imagenet_std[i] + imagenet_mean[i]
    return img

def normalize_image(img):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()  # convert tensor → numpy
    if img.dtype == np.float32 or img.dtype == np.float64:
        return np.clip(img, 0, 1)
    else:
        return np.clip(img, 0, 255).astype(np.uint8)
   
def diceplotpicresult(i, background, prior, rawimage, ground_truth, defect, savepath, save_image=True): 
    if save_image and not os.path.exists(savepath):
        os.makedirs(savepath)

    images = [
        (normalize_image(rawimage), 'Testimage.png'),
        (normalize_image(prior), 'Mask_image.png'),
        (normalize_image(background), 'Background.png'),
        (normalize_image(ground_truth), 'Ground_truth.png'),
        (np.array(Image.fromarray((defect * 255).astype(np.uint8)).convert('L')), 'Anomaly.png')
    ]
    
    # save each image individually
    for img, filename in images:
        plt.figure(figsize=(5, 5))
        if filename == 'Anomaly.png':
            plt.imshow(img, cmap='gray')  
        else:
            plt.imshow(img)   
        plt.axis('off')  

        if save_image:
            file_path = os.path.join(savepath, f"{i+1}_{filename}")
            plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
            plt.close()

    # save all images in a single combined figure
    cols = len(images)
    plt.figure(figsize=(5*cols, 5))
    for idx, (img, filename) in enumerate(images):
        plt.subplot(1, cols, idx+1)
        if filename == 'Anomaly.png':
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)
        plt.title(filename.replace('.png',''))
        plt.axis('off')

    if save_image:
        combined_path = os.path.join(savepath, f"{i+1}_Combined.png")
        plt.savefig(combined_path, bbox_inches='tight', pad_inches=0)
        plt.close()

