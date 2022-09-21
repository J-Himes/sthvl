from tqdm import tqdm
import numpy as np
from glob import glob
import torch
import os

def main():
    feature_dirs = glob('data/charades_pkl/Charades_v1_features_rgb/*', recursive=True)
    new_features_dir = 'data/charades_pkl/Charades_v1_features_rgb_1024_3/'
    if not os.path.exists(new_features_dir):
        os.makedirs(new_features_dir)

    for dir in tqdm(feature_dirs):
        if dir[-3:] == 'csv':
            continue
        new_feature_dir = new_features_dir + dir[-5:]
        if not os.path.exists(new_feature_dir):
            os.makedirs(new_feature_dir)
        path_cmd = dir + '/*'
        video_paths = glob(path_cmd, recursive=True)

        try:
            path_array = video_paths[0].split('/')
        except:
            print(dir)
        save_path = new_feature_dir + '/' + path_array[-1][:5] + '.npy'
        if os.path.exists(save_path):
            continue
        i = 0        
        features = np.zeros((len(video_paths), 1024))
        for path in video_paths:
            fc7 = np.loadtxt(path)
            fc7 = torch.Tensor(fc7).cuda()
            map_vid = torch.nn.Linear(4096, 1024).cuda()
            video = map_vid(fc7)
            features[i] = video.cpu().detach().numpy()
            i += 1            

        with open(save_path, 'wb') as f:
            np.save(f, features)

	
if __name__ == "__main__":
    main()
