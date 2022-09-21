from tqdm import tqdm
import numpy as np
from glob import glob
import pandas as pd
import torch
import os

def main():
    csv_path='../data/Charades_v1_features_rgb/Charades_v1_train.csv'
    csv = pd.read_csv(csv_path)
    word_counts = []
    for description in csv.descriptions:
        word_counts.append(len(description))
    print('Max words: ', csv.descriptions.map(len).max())
    print('Average words: ', np.mean(word_counts))
    print('Std words: ', np.std(word_counts))

    video_counts = []
    for length in csv.length:
        video_counts.append(length * 6)
    print('Max video: ', max(csv.length))
    print('Average video: ', np.mean(video_counts))
    print('Std video: ', np.std(video_counts))

    return
    most_frames = 0
    for dir in tqdm(feature_dirs):
        path_cmd = dir + '/*'
        video_paths = glob(path_cmd, recursive=True)
        for path in video_paths:
            video = np.load(path)
            frames = video.shape[0]
            if frames > most_frames:
                most_frames = frames
    print(most_frames)
'''
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
'''
	
if __name__ == "__main__":
    main()
