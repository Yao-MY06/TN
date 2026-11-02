import numpy as np
import pandas as pd
import os
import cv2
import argparse

def gaussian_kernel(size, variance):
    x, y = np.mgrid[-size:size+1, -size:size+1]
    g = np.exp(-(x**2+y**2)/float(2*variance))
    return g

def create_gaussian(size, variance):
    gaussian_kernel_array = gaussian_kernel(size, variance)
    gaussian_kernel_array =  gaussian_kernel_array * 255/gaussian_kernel_array[int(len(gaussian_kernel_array)/2)][int(len(gaussian_kernel_array)/2)]
    gaussian_kernel_array = gaussian_kernel_array.astype(int)
    return gaussian_kernel_array

def create_gt_images(path_input, path_output, size, variance, width, height, games):
    gaussian_kernel_array = create_gaussian(size, variance)
    for game in games:
        game_dir = os.path.join(path_input, game)
        if not os.path.isdir(game_dir):
            continue
        clips = os.listdir(game_dir)
        for clip in clips:
            print('game = {}, clip = {}'.format(game, clip))

            # heatmaps go under <path_output>/gts/<game>/<clip> to match the
            # 'gt_path' column written by create_gt_labels()
            path_out_game = os.path.join(path_output, 'gts', game)
            if not os.path.exists(path_out_game):
                os.makedirs(path_out_game)

            path_out_clip = os.path.join(path_out_game, clip)    
            if not os.path.exists(path_out_clip):
                os.makedirs(path_out_clip)  

            path_labels = os.path.join(os.path.join(path_input, game, clip), 'Label.csv')
            labels = pd.read_csv(path_labels)    
            for idx in range(labels.shape[0]):
                file_name = labels.loc[idx, 'file name']
                vis = labels.loc[idx, 'visibility']
                x = labels.loc[idx, 'x-coordinate']
                y = labels.loc[idx, 'y-coordinate']
                heatmap = np.zeros((height, width, 3), dtype=np.uint8)
                if vis != 0:
                    x = int(x)
                    y = int(y)
                    for i in range(-size, size+1):
                        for j in range(-size, size+1):
                                if x+i<width and x+i>=0 and y+j<height and y+j>=0 :
                                    temp = gaussian_kernel_array[i+size][j+size]
                                    if temp > 0:
                                        heatmap[y+j,x+i] = (temp,temp,temp)

                cv2.imwrite(os.path.join(path_out_clip, file_name), heatmap) 
                
def create_gt_labels(path_input, path_output, games, train_rate=0.7, split_runs=False):
    """Build labels_train.csv / labels_val.csv.

    split_runs=True  (needed for OpenTTGames-style data)
        Treat the clip as a *concatenation of independent frame runs*. A
        3-frame sample is only formed when all three frames belong to the same
        run, so no sample straddles a temporal gap.
        Requires a `run_id` column in Label.csv (written by prepare_openttgames.py).

    split_runs=False (original tennis behaviour)
        Treat the whole clip as one continuous video.
    """
    df = pd.DataFrame()
    for game in games:
        game_dir = os.path.join(path_input, game)
        if not os.path.isdir(game_dir):
            continue
        clips = os.listdir(game_dir)
        for clip in clips:
            labels = pd.read_csv(os.path.join(path_input, game, clip, 'Label.csv'))
            labels['gt_path'] = 'gts/' + game + '/' + clip + '/' + labels['file name']
            labels['path1'] = 'images/' + game + '/' + clip + '/' + labels['file name']

            if split_runs:
                if 'run_id' not in labels.columns:
                    raise ValueError(
                        f"{os.path.join(path_input, game, clip, 'Label.csv')} has no "
                        f"'run_id' column but split_runs=True was requested")
                parts = []
                # iterate each independent run, mirroring the original slicing
                for _, grp in labels.groupby('run_id', sort=False):
                    grp = grp.reset_index(drop=True)
                    if grp.shape[0] < 3:
                        continue
                    t = grp.iloc[2:].copy()
                    t['path2'] = list(grp['path1'][1:-1])
                    t['path3'] = list(grp['path1'][:-2])
                    parts.append(t)
                if not parts:
                    continue
                labels_target = pd.concat(parts, ignore_index=True)
            else:
                labels_target = labels[2:].copy()
                labels_target['path2'] = list(labels['path1'][1:-1])
                labels_target['path3'] = list(labels['path1'][:-2])

            df = pd.concat([df, labels_target], ignore_index=True)
    df = df.reset_index(drop=True) 
    df = df[['path1', 'path2', 'path3', 'gt_path', 'x-coordinate', 'y-coordinate', 'status', 'visibility']]
    df = df.sample(frac=1)
    num_train = int(df.shape[0]*train_rate)
    df_train = df[:num_train]
    df_test = df[num_train:]
    df_train.to_csv(os.path.join(path_output, 'labels_train.csv'), index=False)
    df_test.to_csv(os.path.join(path_output, 'labels_val.csv'), index=False)  


if __name__ == '__main__':
    # Defaults follow the original TENNIS setup.
    # For table tennis (smaller, faster ball) use: --size 10 --variance 5
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_input', type=str, required=True, help='path to input folder')
    parser.add_argument('--path_output', type=str, required=True, help='path to output folder')
    parser.add_argument('--size', type=int, default=20, help='gaussian kernel radius (tennis=20, TT=8~12)')
    parser.add_argument('--variance', type=int, default=10, help='gaussian variance (tennis=10, TT=4~6)')
    parser.add_argument('--width', type=int, default=1280, help='frame width')
    parser.add_argument('--height', type=int, default=720, help='frame height')
    parser.add_argument('--games', type=str, default='game1,game2,game3,game4,game5,game6,game7,game8,game9,game10',
                        help='comma-separated game folder names')
    parser.add_argument('--train_rate', type=float, default=0.7, help='train/val split ratio')
    parser.add_argument('--split_runs', action='store_true',
                        help='clip = concatenation of independent runs; do not build samples across gaps '
                             '(required for OpenTTGames data)')
    args = parser.parse_args()

    games = [g.strip() for g in args.games.split(',') if g.strip()]

    if not os.path.exists(args.path_output):
        os.makedirs(args.path_output)

    create_gt_images(args.path_input, args.path_output, args.size, args.variance,
                     args.width, args.height, games)
    create_gt_labels(args.path_input, args.path_output, games, args.train_rate,
                     split_runs=args.split_runs)

                            
    
    
    
    