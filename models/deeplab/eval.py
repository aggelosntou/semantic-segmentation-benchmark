"""
DeepLab Evaluation Script

Usage:
  # Full pipeline: predict + evaluate 
  python models/deeplab/eval.py

  # Predict-only (generate segmentations)
  python models/deeplab/eval.py --predict

  # Evaluate-only (use existing segmentations) 
  python models/deeplab/eval.py --evaluate

  # Single-image evaluation with new prediction
  python models/deeplab/eval.py --single-image <split> <image_filename>

  # Single-image metrics-only
  python models/deeplab/eval.py --evaluate --single-image <split> <image_filename>

Options:
  <split>           one of: train, lowres, test
  <image_filename>  exact filename (e.g. tile_42.png) in dataset/<split>/image/
"""
import os
import sys
import argparse
import numpy as np
from glob import glob
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
import torch
from torchvision import transforms as T


SCRIPT_DIR   = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from models.deeplab.dataset import load_dataset, rgb_to_mask, CLASS_RGB
from models.deeplab.model import build_model

CLASS_NAMES = [
    'Unknown','Artificial','Woodland','Arable',
    'Frygana','Bareland','Water','Permanent'
]

WEIGHTS = {
    'train':  'deeplab_train.pth',
    'lowres': 'deeplab_train.pth',
    'test':   'deeplab_train.pth'
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict_split(model, img_dir, out_dir, single_image=None):
    if single_image:
        paths = [os.path.join(img_dir, single_image)]
    else:
        paths = sorted(glob(os.path.join(img_dir, '*.png')) + glob(os.path.join(img_dir, '*.jpg')))
    os.makedirs(out_dir, exist_ok=True)
    inv_map = {v: rgb for rgb, v in CLASS_RGB.items()}
    tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    model.eval()
    with torch.no_grad():
        for p in paths:
            img = Image.open(p).convert('RGB')
            inp = tf(img).unsqueeze(0).to(DEVICE)
            out = model(inp)['out'][0].argmax(0).cpu().numpy()
            h, w = out.shape
            vis = np.zeros((h, w, 3), dtype=np.uint8)
            for cls_idx, rgb in inv_map.items():
                vis[out == cls_idx] = rgb
            name = os.path.basename(p).rsplit('.',1)[0] + '.png'
            Image.fromarray(vis).save(os.path.join(out_dir, name))
    print(f"[INFO] Saved {len(paths)} masks to {out_dir}")


def evaluate_split(pred_dir, img_dir, mask_dir, single_image=None):
    """Compute and return P/R/F1 metrics for one split."""
    if single_image:
        base = os.path.splitext(single_image)[0]
        # GT
        gt_pattern = os.path.join(mask_dir, base + '.*')
        gt_files = glob(gt_pattern)
        if not gt_files:
            raise FileNotFoundError(f"No GT mask for {single_image} in {mask_dir}")
        gt_mask = rgb_to_mask(Image.open(gt_files[0]).convert('RGB'))
        y_true = gt_mask.flatten()
        # Pred
        pred_path = os.path.join(pred_dir, base + '.png')
        if not os.path.isfile(pred_path):
            raise FileNotFoundError(f"No prediction for {single_image} in {pred_dir}")
        pred_mask = rgb_to_mask(Image.open(pred_path).convert('RGB'))
        y_pred = pred_mask.flatten()
    else:
        _, true_masks = load_dataset(img_dir, mask_dir)
        pred_paths = sorted(glob(os.path.join(pred_dir, '*.png')))
        if len(pred_paths) != len(true_masks):
            raise RuntimeError(f"{len(pred_paths)} preds vs {len(true_masks)} GT")
        y_true, y_pred = [], []
        for gt, pf in zip(true_masks, pred_paths):
            y_true.append(gt.flatten())
            pm = rgb_to_mask(Image.open(pf).convert('RGB'))
            y_pred.append(pm.flatten())
        y_true = np.hstack(y_true)
        y_pred = np.hstack(y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(CLASS_NAMES))), zero_division=0
    )
    return {'p':p, 'r':r, 'f1':f1}





def main():
    parser = argparse.ArgumentParser(description="DeepLab pipeline: predict/evaluate with terminal output.")
    parser.add_argument('--predict', action='store_true', help='only generate segmentations')
    parser.add_argument('--evaluate', action='store_true', help='only compute metrics')
    parser.add_argument('--single-image', nargs=2, metavar=('SPLIT','IMAGE'), help='evaluate one image')
    parser.add_argument('--model_path', type=str, help='path to model weights file')
    args = parser.parse_args()
    do_pred = args.predict or not args.evaluate
    do_eval = args.evaluate or not args.predict

    if args.single_image:
        splits = [args.single_image[0]]
    else:
        splits = ['train', 'test', 'lowres']

    results = {}
    for split in splits:
        if args.model_path:
            weights_path = args.model_path
        else:
            weights_path = os.path.join(PROJECT_ROOT, 'scripts', WEIGHTS.get(split, 'deeplab_train.pth'))
        
        if not os.path.exists(weights_path):
            print(f" Model weights not found at: {weights_path}")
            print("Available files in scripts folder:")
            scripts_dir = os.path.join(PROJECT_ROOT, 'scripts')
            if os.path.exists(scripts_dir):
                for f in os.listdir(scripts_dir):
                    if f.endswith('.pth'):
                        print(f"  - {f}")
            continue
            
        print(f"\n=== [{split.upper()}] Loading {os.path.basename(weights_path)} ===")
        model = build_model(DEVICE)
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        model.to(DEVICE)
        
        img_dir = os.path.join(PROJECT_ROOT,'dataset',split,'image')
        mask_dir = os.path.join(PROJECT_ROOT,'dataset',split,'mask')
        if not os.path.isdir(mask_dir): mask_dir = os.path.join(PROJECT_ROOT,'dataset',split,'masks')
        pred_dir = os.path.join(PROJECT_ROOT,'predictions','deeplab',split)
        
        if args.single_image:
            if split == args.single_image[0]:
                if do_pred:
                    predict_split(model,img_dir,pred_dir,single_image=args.single_image[1])
                if do_eval:
                    m = evaluate_split(pred_dir,img_dir,mask_dir,single_image=args.single_image[1])
                    print(f"\n Metrics for '{args.single_image[1]}' in {split}:")
                    print(f"{'Class':<12}{'P':>6}{'R':>6}{'F1':>6}")
                    print('-'*30)
                    for i,n in enumerate(CLASS_NAMES): 
                        print(f"{n:<12}{m['p'][i]:6.3f}{m['r'][i]:6.3f}{m['f1'][i]:6.3f}")
                return
        else:
            if do_pred:
                predict_split(model,img_dir,pred_dir)
            if do_eval:
                print(f"Evaluating '{split}'")
                res = evaluate_split(pred_dir,img_dir,mask_dir)
                results[split] = res

                print(f"\n Metrics for '{split}':")
                print(f"{'Class':<12}{'P':>6}{'R':>6}{'F1':>6}")
                print('-'*30)
                for i,n in enumerate(CLASS_NAMES): 
                    print(f"{n:<12}{res['p'][i]:6.3f}{res['r'][i]:6.3f}{res['f1'][i]:6.3f}")
    
    if len(results) > 1:
        print(f"\n SUMMARY:")
        for split, res in results.items():
            print(f"{split}: P={res['p'].mean():.3f}, R={res['r'].mean():.3f}, F1={res['f1'].mean():.3f}")

if __name__ == '__main__':
    main()
