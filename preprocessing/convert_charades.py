from tqdm import tqdm
import numpy as np
from glob import glob
import torch
import os

def main():
    feature_dirs = glob('data/Charades_v1_features_rgb/*', recursive=True)
    new_features_dir = 'data/Charades_v1_features_rgb_1024/'
    if not os.path.exists(new_features_dir):
        os.makedirs(new_features_dir)

    for dir in tqdm(feature_dirs):
        new_feature_dir = new_features_dir + dir[-5:]
        if not os.path.exists(new_feature_dir):
            os.makedirs(new_feature_dir)
        path_cmd = dir + '/*'
        video_paths = glob(path_cmd, recursive=True)

        for path in video_paths:
            path_array = path.split('/')
            save_path = new_feature_dir + '/' + path_array[-1][:-4] + '.npy'
            if os.path.exists(save_path):
                continue

            fc7 = np.loadtxt(path)
            fc7 = torch.Tensor(fc7).cuda()
            map_vid = torch.nn.Linear(4096, 1024).cuda()
            video = map_vid(fc7)
            with open(save_path, 'wb') as f:
                np.save(f, video.cpu().detach().numpy())

if __name__ == "__main__":
    main()