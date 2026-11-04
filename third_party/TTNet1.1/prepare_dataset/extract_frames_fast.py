# -*- coding: utf-8 -*-
"""
Optimized version of extract_smooth_labellings.py.
Same frame-selection logic (empty_event +/-4, other events +/-8), but:
  - removes cv2.waitKey() (the main bottleneck in headless batch mode)
  - prints per-video progress
  - skips already-extracted frames (resumable)
"""
import os
import json
import time
from glob import glob

import cv2


def make_folder(folder_path):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)


def get_frame_indexes(events_annos_path, num_frames_from_event):
    with open(events_annos_path) as f:
        events_annos = json.load(f)
    selected_indexes = []
    for main_f_idx in events_annos:
        event = events_annos[main_f_idx]
        main_f_idx = int(main_f_idx)
        if event == 'empty_event':  # +/-4
            for idx in range(main_f_idx - num_frames_from_event, main_f_idx + num_frames_from_event + 1):
                selected_indexes.append(idx)
        else:  # +/-8
            for idx in range(main_f_idx - num_frames_from_event * 2, main_f_idx + num_frames_from_event * 2 + 1):
                selected_indexes.append(idx)
    return set(selected_indexes)


def extract_images_from_videos(video_path, events_annos_path, out_images_dir, num_frames_from_event):
    selected_indexes = get_frame_indexes(events_annos_path, num_frames_from_event)
    video_fn = os.path.basename(video_path)[:-4]
    sub_images_dir = os.path.join(out_images_dir, video_fn)
    make_folder(sub_images_dir)

    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('[{}] total frames: {}, selected: {}'.format(video_fn, n_frames, len(selected_indexes)))

    frame_cnt = -1
    saved = 0
    skipped = 0
    t0 = time.time()
    while True:
        ret, img = cap.read()
        if not ret:
            break
        frame_cnt += 1
        if frame_cnt in selected_indexes:
            image_path = os.path.join(sub_images_dir, 'img_{:06d}.jpg'.format(frame_cnt))
            if os.path.isfile(image_path):
                skipped += 1
            else:
                cv2.imwrite(image_path, img)
                saved += 1
        if frame_cnt % 2000 == 0 and frame_cnt > 0:
            print('    {}: processed {} frames, saved {}'.format(video_fn, frame_cnt, saved))
    cap.release()
    print('    done {}: saved {}, skipped {}, {:.1f}s'.format(video_fn, saved, skipped, time.time() - t0))


if __name__ == '__main__':
    dataset_dir = '../dataset'
    num_frames_sequence = 9
    num_frames_from_event = int((num_frames_sequence - 1) / 2)

    for dataset_type in ['training', 'test']:
        video_dir = os.path.join(dataset_dir, dataset_type, 'videos')
        annos_dir = os.path.join(dataset_dir, dataset_type, 'annotations')
        out_images_dir = os.path.join(dataset_dir, dataset_type, 'images')

        video_paths = sorted(glob(os.path.join(video_dir, '*.mp4')))
        # exclude demo_input.mp4 if present
        video_paths = [p for p in video_paths if os.path.basename(p) != 'demo_input.mp4']

        for video_path in video_paths:
            video_fn = os.path.basename(video_path)[:-4]
            events_annos_path = os.path.join(annos_dir, video_fn, 'events_markup.json')
            if not os.path.isfile(events_annos_path):
                print('skip {}: no events_markup.json'.format(video_fn))
                continue
            extract_images_from_videos(video_path, events_annos_path, out_images_dir, num_frames_from_event)

    print('=== ALL EXTRACTION DONE ===')
