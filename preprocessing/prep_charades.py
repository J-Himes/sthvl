import math
import os
from torch.utils.data import Dataset
import numpy as np
import csv
import pandas as pd
from collections import defaultdict
import json
import random
import math
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
    # inputs
    csv_path = '../data/Charades_v1_features_rgb/Charades_v1_train.csv'
    json_path = '../data/charades/charades.json'
    features_path = '../data/Charades_v1_features_rgb_1024'
    feature_framerate = 1
    max_words = 32
    max_frames = 32
    summ_type = None
    split_type = 'train'

    # start of old dataloader
    csv = pd.read_csv(csv_path)
    data = json.load(open(json_path, 'r'))
    path_cmd = features_path + '/*'
    feature_dict = glob(path_cmd, recursive = False)
    feature_framerate = feature_framerate
    max_words = max_words
    max_frames = max_frames
    summ_type=summ_type

    feature_size = 1024

    assert split_type in ["train", "val", "test"]
    video_ids = [feature_dict[idx][-5:] for idx in range(len(feature_dict))]
    feature_dict = dict(zip(video_ids, feature_dict))
    choiced_video_ids = []
    for key, value in data.items():
        if value['subset'][:len(split_type)] == split_type:
            choiced_video_ids.append(key)

    labels, pop_list = make_dataset(data, choiced_video_ids, feature_dict, num_classes=157)
    for key in pop_list:
        choiced_video_ids.pop(key[0])

    sample_len = 0
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

    sample_len = len(sentences_dict)

    # new data
    for idx in sentences_dict:
        video_id, caption = sentences_dict[idx]
        label = labels[idx][1]
        load_path = feature_dict[video_id] + '/' + feature_dict[video_id][-5:] + '.npy'
        video = np.load(load_path)

        iterations = math.ceil(len(video) / max_frames)
        for it in range(1, iterations+1):
            idx1 = (it-1)*max_frames
            idx2 = it*max_frames
            video_slice = video[idx1:idx2, :]
            label_slice = label[:, idx1:idx2]
            idx1m = idx1 % max_frames
            caption_slice = caption[idx1m*max_frames:it*max_frames]


if __name__ == "__main__":
    main()
