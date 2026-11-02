"""TrackNet reproduction evaluator.

Runs the pretrained checkpoint over the validation split and reports the
repo-standard metrics (precision / recall / F1, min_dist=5) PLUS the accuracy
metric used in the original TrackNet paper, broken down by visibility class.

Metrics definition follows general.validate() exactly:
  - TP/FP/FN/TN are accumulated per visibility class (1,2,3); index 0 unused.
  - A prediction is correct when a ball is detected AND euclidean(pred, gt) < min_dist.
  - Precision = TP / (TP + FP)          (all classes)
  - Recall    = TP / (class1+2+3 totals)
  - F1        = 2PR/(P+R)
  - Accuracy  = (TP + TN) / (TP + TN + FP + FN)

Usage (run from the repo root that contains ./datasets/trackNet):
    python evaluate.py --model_path weights/model_best.pt --batch_size 4
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import distance

from model import BallTrackerNet
from datasets import trackNetDataset
from general import postprocess


def evaluate(model, loader, device, min_dist=5, out_height=360, out_width=640,
             min_radius=2, max_radius=7, verbose=False, amp=False):
    model.eval()
    crit = nn.CrossEntropyLoss()
    losses = []
    tp = [0, 0, 0, 0]
    fp = [0, 0, 0, 0]
    tn = [0, 0, 0, 0]
    fn = [0, 0, 0, 0]
    use_amp = amp and device == "cuda"

    for it, batch in enumerate(loader):
        with torch.no_grad():
            x = batch[0].float().to(device)
            if use_amp:
                with torch.autocast("cuda", dtype=torch.float16):
                    out = model(x)
                    gt = torch.as_tensor(batch[1], dtype=torch.long, device=device)
                    loss = crit(out, gt)
                out = out.float()
            else:
                out = model(x)
                gt = torch.as_tensor(batch[1], dtype=torch.long, device=device)
                loss = crit(out, gt)
            losses.append(loss.item())

            output = out.argmax(dim=1).detach().cpu().numpy()
            for i in range(len(output)):
                x_pred, y_pred = postprocess(output[i], height=out_height, width=out_width,
                                             min_radius=min_radius, max_radius=max_radius)
                x_gt, y_gt, vis = batch[2][i], batch[3][i], batch[4][i]
                if x_pred:
                    if vis != 0:
                        dst = distance.euclidean((x_pred, y_pred), (x_gt, y_gt))
                        if dst < min_dist:
                            tp[vis] += 1
                        else:
                            fp[vis] += 1
                    else:
                        fp[vis] += 1
                else:
                    if vis != 0:
                        fn[vis] += 1
                    else:
                        tn[vis] += 1
        if (it + 1) % 25 == 0:
            print(f"  iter {it+1}/{len(loader)}  loss={np.mean(losses):.6f}  "
                  f"tp={sum(tp)} tn={sum(tn)} fp={sum(fp)} fn={sum(fn)}", flush=True)
    eps = 1e-15
    precision = sum(tp) / (sum(tp) + sum(fp) + eps)
    vc1 = tp[1] + fp[1] + tn[1] + fn[1]
    vc2 = tp[2] + fp[2] + tn[2] + fn[2]
    vc3 = tp[3] + fp[3] + tn[3] + fn[3]
    recall = sum(tp) / (vc1 + vc2 + vc3 + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    correct = sum(tp) + sum(tn)
    total = sum(tp) + sum(tn) + sum(fp) + sum(fn)
    accuracy = correct / (total + eps)

    per_class = {}
    for c in (1, 2, 3):
        p = tp[c] / (tp[c] + fp[c] + eps)
        r = tp[c] / (tp[c] + fn[c] + eps)
        per_class[f"visibility_{c}"] = {
            "tp": tp[c], "fp": fp[c], "fn": fn[c], "tn": tn[c],
            "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r + eps), 4),
        }

    return {
        "val_loss": round(float(np.mean(losses)), 6),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "accuracy": round(float(accuracy), 4),
        "tp": int(sum(tp)), "tn": int(sum(tn)), "fp": int(sum(fp)), "fn": int(sum(fn)),
        "n_samples": int(total),
        "min_dist": min_dist,
        "per_class": per_class,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True, help="path to checkpoint")
    ap.add_argument("--dataset_path", type=str, default="./datasets/trackNet",
                    help="root of the prepared dataset (must contain labels_val.csv)")
    ap.add_argument("--input_height", type=int, default=360)
    ap.add_argument("--input_width", type=int, default=640)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--min_dist", type=int, default=5,
                    help="hit radius in the 640x360 output space (tennis=5; "
                         "for table tennis the ball is ~16px wide, 8-10 is reasonable)")
    ap.add_argument("--min_radius", type=int, default=2,
                    help="HoughCircles min radius (tennis=2)")
    ap.add_argument("--max_radius", type=int, default=7,
                    help="HoughCircles max radius (tennis=7; for table tennis the ball is "
                         "~16px wide in the 640x360 output, so ~12)")
    ap.add_argument("--out", type=str, default="eval_results.json")
    ap.add_argument("--amp", action="store_true", help="fp16 inference (much faster)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")
    if device == "cuda":
        print(f"gpu    = {torch.cuda.get_device_name(0)}")

    val_ds = trackNetDataset("val", args.input_height, args.input_width, args.dataset_path)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = BallTrackerNet()
    state = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)

    res = evaluate(model, val_loader, device, min_dist=args.min_dist,
                   out_height=args.input_height, out_width=args.input_width,
                   min_radius=args.min_radius, max_radius=args.max_radius,
                   amp=args.amp)
    res["dataset_path"] = args.dataset_path
    res["model_path"] = args.model_path
    print("\n================ RESULTS ================")
    print(json.dumps(res, indent=2, ensure_ascii=False))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\nsaved -> {os.path.abspath(args.out)}")
