import time
from datetime import datetime
from util import *
import argparse
import numpy as np
import torch
import os
import random
import statistics

# Google Drive file IDs for each pretrained checkpoint.
# Fill in the IDs for the models you have links for; leave None to skip auto-download.
_MODEL_GDRIVE_IDS = {
    '01':       '1ZzkC_F_qTWgoAf5NxB7Y52uct0y6jSRW',
    '02':       '17M9brXW0Ps3uYz3rh0DwtnAlAn816NDX',
    'hazelnut': '1khd6y3NfViZJrd9PcZiJxPjieCYug3Hp',
}

def ensure_model(data: str) -> str:
    """Return checkpoint path, downloading from Google Drive if the file is missing."""
    os.makedirs('./models', exist_ok=True)
    path = f'./models/{data}-anosups.pth'
    if os.path.exists(path):
        return path

    file_id = _MODEL_GDRIVE_IDS.get(data)
    if file_id is None:
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            f"No download URL configured for dataset '{data}'. "
            "Please add the Google Drive file ID to _MODEL_GDRIVE_IDS."
        )

    try:
        import gdown
    except ImportError:
        raise ImportError("Run `pip install gdown` to enable automatic model download.")

    url = f'https://drive.google.com/uc?id={file_id}'
    print(f"Model not found at {path}. Downloading from Google Drive...")
    gdown.download(url, path, quiet=False)
    if not os.path.exists(path):
        raise RuntimeError(f"Download failed - file not found after download: {path}")
    print(f"Model saved to {path}.")
    return path

parser = argparse.ArgumentParser(description="cycle")
parser.add_argument('--data', type=str, default="02", help='data type')
parser.add_argument('--anomaly', type=str, default="line", help='anomaly type')
parser.add_argument('--a', type=float, default=0.06, help='the threold of sups step')
parser.add_argument('--b', type=float, default=0.35, help='the dice threold of step2')
parser.add_argument('--c', type=float, default=0.35, help='the iou threold of step2')
parser.add_argument('--degree', type=str, default="big", help='size of anomaly ')
args, unknown = parser.parse_known_args()
_RUN_TS = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def results_root() -> str:
    folder = os.path.splitext(os.path.basename(__file__))[0] + '_' + _RUN_TS
    return os.path.join('.', 'results', folder)

def get_hyper_paras(class_name, anomaly):
    if class_name == '02':
        paras_dict = {'line': {'threshold': (0.06, 0.35, 0.35), 'degree': ('big', 'small')},
                      'hole': {'threshold': (0.06, 0.35, 0.35), 'degree':('1', '3', '5')},
                      'color': {'threshold': (0.06, 0.11, 0.11), 'degree': ('big', 'small')}}

    elif class_name == '01':
        paras_dict = {'line': {'threshold': (0.95, 0.44, 0.44), 'degree': ('big', 'small')},
                      'hole': {'threshold': (1.1, 0.30, 0.360), 'degree':('1', '3', '5')},
                      'color': {'threshold': (1, 0.345, 0.345), 'degree': ('big', 'small')}}

    elif class_name == 'hazelnut':
        paras_dict = {'crack': {'threshold': (0.034, 0.02, 0.02)},
                      'cut': {'threshold': (0.046, 0.023, 0.023)},
                      'print': {'threshold': (0.06, 0.025, 0.025)},
                      'hole': {'threshold': (0.06, 0.025, 0.025)}}
    else:
        raise NotImplementedError('Please manually define these hyper-parameters')
    return paras_dict[anomaly]


def main():
    chkpt_dir = ensure_model(args.data)
    model_ano = prepare_model(chkpt_dir, 'mae_vit_large_patch16')
    print('Model loaded.')

    imgs, gtimgs1, _ = pre_data(args.anomaly, args.data, args.degree, base_dir="./images")
    savepath = os.path.join(results_root(), args.data, args.anomaly, args.degree)
    os.makedirs(savepath, exist_ok=True)

    seed = 2025
    torch.manual_seed(seed)
    random.seed(seed)

    results_dict = {}
    start = time.time()

    repeat = 1
    for i, img in enumerate(imgs):
        dice_list, iou_list = [], []

        for r in range(repeat):
            ssimindex11 = getsim(196,img, model_ano, 0.5, args.a)
            original, reconstruction, maskimg = run_one_image(img, model_ano, np.array(ssimindex11), 0)

            prior = post_process(reconstruction)
            maskimg = post_process(maskimg)
            rawimage = post_process(original)
            ground_truth = post_process(gtimgs1[i])

            defect_b = getdefect(rawimage, prior, args.b)
            dice = dice_coefficient(ground_truth, defect_b)

            defect_c = getdefect(rawimage, prior, args.c)
            iou = iou_coefficient(ground_truth, defect_c)

            dice_list.append(dice)
            iou_list.append(iou)

        dice_avg = float(np.mean(dice_list))
        iou_avg  = float(np.mean(iou_list))

        print(f"[{i}] Dice: {dice_avg:.4f} (repeat {repeat}-avg), "
              f"IoU: {iou_avg:.4f} (repeat {repeat}-avg)")

        results_dict[f'results{i + 1}'] = {
            'dice': dice_avg,
            'iou': iou_avg
        }

        #diceplotpicresult(i, prior, maskimg, rawimage, ground_truth, defect_c, savepath, save_image=True)
    total_time = time.time() - start

    mean_dice = sum(v['dice'] for v in results_dict.values()) / len(results_dict)
    mean_iou = sum(v['iou'] for v in results_dict.values()) / len(results_dict)
    std_dice = statistics.stdev(v['dice'] for v in results_dict.values())
    std_iou = statistics.stdev(v['iou'] for v in results_dict.values())
    
    with open(os.path.join(savepath, f'{args.data}_{args.anomaly}_{args.degree}.txt'), 'w') as f:
        for key, value in results_dict.items():
            f.write(f"[{key}] Dice: {value['dice']:.4f}, IoU: {value['iou']:.4f}\n")

    print(f'Total time: {total_time:.2f}s, Images: {len(imgs)}, Mean Dice: {mean_dice:.4f}, Mean IoU: {mean_iou:.4f}')

    with open(os.path.join(results_root(), f'summary_{_RUN_TS}.txt'), 'a') as file:
        line = ' '.join(map(str, [
            args, 'total_time:', total_time,
            'no_of_images:', len(imgs),
            'mean_dice:', mean_dice,
            'mean_iou:', mean_iou,
            'std_dice:', std_dice,
            'std_iou:', std_iou
        ]))
        file.write(line + '\n')

if __name__ == "__main__":
    data_anomaly_map = {
        '01': ('line', 'color', 'hole'),
        '02': ('line', 'color', 'hole'),
        'hazelnut': ('crack', 'cut', 'print', 'hole')
    }
    
    if args.data not in data_anomaly_map:
        raise ValueError(f"Unsupported data type: {args.data}. Supported types: {list(data_anomaly_map.keys())}")
    
    anomalies_list = data_anomaly_map[args.data]
    
    for anomaly in anomalies_list:
        args.anomaly = anomaly
        hyper_paras = get_hyper_paras(args.data, args.anomaly)
        args.a, args.b, args.c = hyper_paras['threshold']
        
        if 'degree' in hyper_paras:
            degrees = hyper_paras['degree']
            for degree in degrees:
                args.degree = degree
                print(args)
                main()
        else:
            args.degree = 'no'
            print(args)
            main()