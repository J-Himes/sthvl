import math
import os
from torch.utils.data import Dataset
import numpy as np
import csv
import pandas as pd
from collections import defaultdict
import json
import random
from tqdm import tqdm
import math
import pickle
from glob import glob

def make_dataset(data, choiced_video_ids, feature_dict, num_classes=157):
    dataset = []

    i = 0
    pop_list = []
    for vid in choiced_video_ids:

        load_path = feature_dict[vid] + '/' + feature_dict[vid][-5:] + '.npy'
        try:
            video_slice = np.load(load_path)
        except:
            pop_list.append([i, vid])
            i += 1
            continue
        num_frames = len(video_slice)

        label = np.zeros((num_classes, num_frames), np.float32)

        fps = num_frames / data[vid]['duration']
        for ann in data[vid]['actions']:
            for fr in range(0, num_frames, 1):
                if fr / fps > ann[1] and fr / fps < ann[2]:
                    label[ann[0], fr] = 1  # binary classification
        dataset.append((vid, label, data[vid]['duration'], num_frames))
        i += 1

    return dataset, pop_list

def main():
    split_types = ['train', 'test']
    csv_paths = {'train': 'data/Charades_v1_features_rgb/Charades_v1_train.csv',
                'test': 'data/Charades_v1_features_rgb/Charades_v1_test.csv'}
    json_path = 'data/charades/charades.json'
    features_path = '../util/data/Charades_v1_features_rgb_1024'
    max_words = 1000
    max_frames = 1000
    output_dir = 'data/Charades_v1_features_rgb_1024_trim' + '_' + str(max_frames) + '_' + str(max_words)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    summ_type = None

    for split_type in split_types:
        csv_path = csv_paths[split_type]
        csv = pd.read_csv(csv_path)
        data = json.load(open(json_path, 'r'))
        path_cmd = features_path + '/*'
        feature_dict = glob(path_cmd, recursive = False)

        video_ids = [feature_dict[idx][-5:] for idx in range(len(feature_dict))]
        feature_dict = dict(zip(video_ids, feature_dict))
        choiced_video_ids = []
        for key, value in data.items():
            if value['subset'][:len(split_type)] == split_type:
                choiced_video_ids.append(key)
        labels, pop_list = make_dataset(data, choiced_video_ids, feature_dict, num_classes=157)
        for key in pop_list:
            choiced_video_ids.pop(key[0])
        sentences_dict = {}
        video_sentences_dict = defaultdict(list)
        i = 0
        for key in choiced_video_ids:
            csv_value = csv.loc[csv['id'] == key]
            try:
                description = csv_value['descriptions'].values[0]
            except:
                continue
            sentences_dict[i] = (key, description)
            video_sentences_dict[key].append(description)
            i += 1

        output_dir2 = output_dir + '/' + split_type
        if not os.path.exists(output_dir2):
            os.makedirs(output_dir2)
        new_csv = []
        for idx in tqdm(sentences_dict):
            video_id, caption = sentences_dict[idx]
            label = labels[idx][1]
            load_path = feature_dict[video_id] + '/' + feature_dict[video_id][-5:] + '.npy'
            video = np.load(load_path)

            iterations = math.ceil(len(video) / max_frames)
            for it in range(1, iterations+1):
                new_key = feature_dict[video_id][-5:] + str(it-1)
                save_path = output_dir2 + '/' + new_key + '.pkl'
                new_csv.append(save_path)

                idx1 = (it-1)*max_frames
                idx2 = it*max_frames
                video_slice = video[idx1:idx2, :]
                label_slice = label[:, idx1:idx2]
                caption_slice = caption[:max_words]
                with open(save_path, 'wb') as f:
                    pickle.dump((new_key, video_slice, label_slice, caption_slice), f)
                # with open(save_path, 'rb') as f:
                #     temp = pickle.load(f)

        csv_save = output_dir + '/' + split_type + '.pkl'
        with open(csv_save, 'wb') as f:
            pickle.dump(new_csv, f)

if __name__ == "__main__":
    main()
