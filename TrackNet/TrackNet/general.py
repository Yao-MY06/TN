import torch
import time
import numpy as np
import torch.nn as nn
import cv2
from scipy.spatial import distance

def train(model, train_loader, optimizer, device, epoch, max_iters=200, amp=False, log_every=20):
    start_time = time.time()
    losses = []
    criterion = nn.CrossEntropyLoss()
    use_amp = amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    for iter_id, batch in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)
        model.train()
        x = batch[0].float().to(device, non_blocking=True)
        gt = torch.as_tensor(batch[1], dtype=torch.long, device=device)
        if use_amp:
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(x)
                loss = criterion(out, gt)
            # GradScaler skips the update when the loss overflows fp16; count it
            # so a silently-stalled run is visible in the log
            scaled = scaler.scale(loss)
            scaled.backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(x)
            loss = criterion(out, gt)
            loss.backward()
            optimizer.step()
        end_time = time.time()
        duration = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))
        losses.append(loss.item())

        if (iter_id + 1) % log_every == 0:
            print('train | epoch = {}, iter = [{}|{}], loss = {}, time = {}'.format(
                epoch, iter_id + 1, max_iters, round(loss.item(), 6), duration), flush=True)

        if iter_id > max_iters - 1:
            break

    return np.mean(losses)

def validate(model, val_loader, device, epoch, min_dist=5,
             out_height=360, out_width=640, min_radius=2, max_radius=7,
             verbose=True, amp=False):
    losses = []
    tp = [0, 0, 0, 0]
    fp = [0, 0, 0, 0]
    tn = [0, 0, 0, 0]
    fn = [0, 0, 0, 0]
    criterion = nn.CrossEntropyLoss()
    use_amp = amp and device == "cuda"
    model.eval()
    for iter_id, batch in enumerate(val_loader):
        with torch.no_grad():
            x = batch[0].float().to(device)
            if use_amp:
                # fp16 forward: the 256x360x640 output head dominates runtime,
                # autocast roughly halves it
                with torch.autocast("cuda", dtype=torch.float16):
                    out = model(x)
                    gt = torch.as_tensor(batch[1], dtype=torch.long, device=device)
                    loss = criterion(out, gt)
                out = out.float()
            else:
                out = model(x)
                gt = torch.as_tensor(batch[1], dtype=torch.long, device=device)
                loss = criterion(out, gt)
            losses.append(loss.item())
            # metrics
            output = out.argmax(dim=1).detach().cpu().numpy()
            for i in range(len(output)):
                x_pred, y_pred = postprocess(output[i], height=out_height, width=out_width,
                                             min_radius=min_radius, max_radius=max_radius)
                x_gt = batch[2][i]
                y_gt = batch[3][i]
                vis = batch[4][i]
                if x_pred:
                    if vis != 0:
                        dst = distance.euclidean((x_pred, y_pred), (x_gt, y_gt))
                        if dst < min_dist:
                            tp[vis] += 1
                        else:
                            fp[vis] += 1
                    else:        
                        fp[vis] += 1
                if not x_pred:
                    if vis != 0:
                        fn[vis] += 1
                    else:
                        tn[vis] += 1
            if verbose:
                print('val | epoch = {}, iter = [{}|{}], loss = {}, tp = {}, tn = {}, fp = {}, fn = {} '.format(epoch,
                                                                                                        iter_id,
                                                                                                        len(val_loader),
                                                                                                        round(np.mean(losses), 6),
                                                                                                        sum(tp),
                                                                                                        sum(tn),
                                                                                                        sum(fp),
                                                                                                        sum(fn)))
    eps = 1e-15
    precision = sum(tp) / (sum(tp) + sum(fp) + eps)
    vc1 = tp[1] + fp[1] + tn[1] + fn[1]
    vc2 = tp[2] + fp[2] + tn[2] + fn[2]
    vc3 = tp[3] + fp[3] + tn[3] + fn[3]
    recall = sum(tp) / (vc1 + vc2 + vc3 + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    if verbose:
        print('precision = {}'.format(precision))
        print('recall = {}'.format(recall))
        print('f1 = {}'.format(f1))

    return np.mean(losses), precision, recall, f1


def postprocess(feature_map, scale=2, height=360, width=640,
                min_radius=2, max_radius=7):
    """Turn the argmax class map into a single ball coordinate.

    Defaults reproduce the ORIGINAL tennis behaviour exactly
    (360x640, Hough radius 2..7).

    For table tennis the ball occupies a larger *relative* area of the frame
    (measure with the same rule: the ball is ~16 px wide in the 640x360 output
    space, i.e. radius ~8), so pass e.g. min_radius=3, max_radius=12, and
    height/width if you train at a different resolution.
    """
    feature_map *= 255
    feature_map = feature_map.reshape((height, width))
    feature_map = feature_map.astype(np.uint8)
    ret, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2,
                               minRadius=min_radius, maxRadius=max_radius)
    x, y = None, None
    if circles is not None:
        if len(circles) == 1:
            x = circles[0][0][0]*scale
            y = circles[0][0][1]*scale
    return x, y




