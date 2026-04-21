from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
from PIL import Image
import numpy as np
import torch


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def pre_data(error, data, degree, base_dir: str = "./images"):
    imgs = []
    gtimgs = []

    base_path = f'{base_dir}/{data}/test/{error}/'
    img_types = ['imgs', 'gt']
    
    
    for img_type in img_types:
        path = f'{base_path}{img_type}/' if data == 'hazelnut' else f'{base_path}{img_type}/{degree}/'
        image_dir = sorted(os.listdir(path))
        
        for file in image_dir:
            if not (file.lower().endswith('.png') or file.lower().endswith('.bmp')):
                continue

            image = Image.open(os.path.join(path, file)).resize((224, 224), Image.Resampling.LANCZOS).convert('RGB')
            img = np.array(image) / 255.0

            assert img.shape == (224, 224, 3)
            img = (img - imagenet_mean) / imagenet_std

            if img_type == 'imgs':
                imgs.append(img)
            elif img_type == 'gt':
                img_tensor = torch.tensor(img).permute(2, 0, 1)
                gtimgs.append(img_tensor.view(1, 3, 224, 224).cpu().data)

    gtimgs = torch.stack(gtimgs)
    return imgs, gtimgs, base_path
