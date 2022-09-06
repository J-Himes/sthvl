from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from torch.utils.data import Dataset
from vsumm.vsumm import vsumm
from vsumm.vsumm_skim import vsumm_skim
import pandas as pd
import os
import numpy as np
import random
import pickle
import mpu
import csv


def _get_text(video_id, n_pair_max, sub_ids=None, ):
    sample_data_dict = data_dict[video_id]

    k = len(sub_ids)
    r_ind = sub_ids
    starts = np.zeros(k)
    ends = np.zeros(k)

    for i in range(k):
        ind = r_ind[i]
        words, start_, end_ = _get_single_transcript(sample_data_dict, ind, )
        starts[i], ends[i] = start_, end_

    return words, starts, ends

def _get_single_transcript(data_dict, ind, ):
    start, end = ind, ind
    words = str(data_dict['text'][ind])
    return words, data_dict['start'][start], data_dict['end'][end]

def _expand_video_slice(s, e, si, ei, fps, video_features):
    start = int(s[si] * fps)
    end = int(e[ei] * fps) + 1

    if start > end:
        start, end = end, start
    video_slice = video_features[start:end]

    expand_left = True
    while len(video_slice) < 1:
        if si==0 and ei==len(s)-1:
            break
        if expand_left:
            expand_left = False
            si = si-1 if si>0 else si
        else:
            expand_left = True
            ei = ei+1 if ei<len(e)-1 else ei
        start = int(s[si] * fps)
        end = int(e[ei] * fps) + 1
        if start > end:
            start, end = end, start
        video_slice = video_features[start:end]

    return video_slice, start, end

def _get_video(idx, s, e, words, only_sim=False):
    samp_videos, samp_s, samp_e, samp_text, samp_id_file, samp_id, slice_shape = [], [], [], [], [], [], []
    features_path = './data/howto100m/howto100m_s3d_features'

    summ_type = None
    feature_framerate = 1
    max_video_length = [0] * len(s)

    feature_file = os.path.join(features_path, howto100_csv["feature_file"].values[idx])
    # try:
    video_features = np.load(feature_file)

    total_frames = 0
    for i in range(len(s)):
        if len(video_features) < 1:
            raise ValueError("{} is empty.".format(feature_file))
        video_slice, start, end = _expand_video_slice(s, e, i, i, feature_framerate, video_features)

        # percent = 50
        # indices = list(range(len(video_slice)))
        # if summ_type == 'sampling':
        #     frames = len(video_slice)
        #     selected_frames = np.arange(frames)
        #     if percent != 75:
        #         rate = percent / 100
        #         selected_frames = selected_frames[
        #             selected_frames * rate == (selected_frames * rate).astype(int)]
        #     else:
        #         rate = 0.25
        #         selected_frames = selected_frames[
        #             selected_frames * rate == (selected_frames * rate).astype(int)]
        #         selected_frames = np.delete(indices, selected_frames)
        # elif summ_type == 'vsumm_key':
        #     selected_frames = vsumm(video_slice, 1, percent)
        # elif summ_type == 'vsumm_skim':
        #     selected_frames = vsumm_skim(video_slice, 1, percent)
        # if summ_type != None:
        #     deleted_frames = [x for x in indices if x not in selected_frames]
        #     video_slice = np.delete(video_slice, deleted_frames, axis=0)


        # max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_shape[0] else slice_shape[0]
        # if len(video_slice) < 1:
        #     pass
        # else:

        video_clip = video_slice

        id_file = howto100_csv["feature_file"].values[idx] # [:11] + '_' + str(i) + howto100_csv["feature_file"].values[idx][11:]
        new_id = howto100_csv["feature_file"].values[idx][:11] #  + '_' + str(i)

        total_frames += video_clip.shape[0]
        # slice_shape.append(video_clip.shape[0])

        print(feature_file)
        print(id_file)
        print(new_id)
        print(video_clip.shape)
        print(words)
        print()

        # organize data to be loaded into csv and pickle file for pretraining
        samp_videos.append(video_clip)
        # samp_s.append(num_frame_start)
        # samp_e.append(num_frame_end)
        # samp_id_file.append(id_file)
        # samp_id.append(new_id)
        samp_text.append(words)


    video = np.zeros((total_frames, 1024), dtype=np.float)
    video[:samp_videos[0].shape[0]] = samp_videos[0]

    idx_frame_start = 0
    idx_frame_end = samp_videos[0].shape[0]
    samp_s.append(idx_frame_start)
    samp_e.append(idx_frame_end)

    for i in range(1, len(samp_videos)):
        idx_frame_start = idx_frame_end
        idx_frame_end += samp_videos[i].shape[0]
        samp_s.append(idx_frame_start)
        samp_e.append(idx_frame_end)

        video[idx_frame_start:idx_frame_end] = samp_videos[i]

    # except Exception as e:
    #     print("video_id: {} error.".format(feature_file))

    return video, id_file, new_id, samp_s, samp_e, samp_text

    def second_to_stamp(self, in_seconds):
        m, s = divmod(in_seconds, 60)
        h, m2 = divmod(m, 60)
        return "%02d:%02d:%02d" % (h, m2, s)

if __name__ == "__main__":
    n_pair, positive_n_pair = 3, 3

    howto100_csv = pd.read_csv('./data/howto100m/debug/HowTo100M_one_thousandth.csv')
    # Get iterator video ids
    video_id_list = [itm for itm in howto100_csv['video_id'].values]
    data_dict = pickle.load(open('./data/howto100m/debug/caption_one_thousandth.pickle', 'rb'))

    video_id2idx_dict = {video_id: id for id, video_id in enumerate(video_id_list)}
    # Get all captions
    iter2video_pairs_dict = {}
    iter2video_pairslist_dict = {}
    iter_idx_mil_ = 0
    for video_id in video_id_list:
        sample_data_dict = data_dict[video_id]
        n_caption = len(sample_data_dict['start'])

        sub_list = []
        if n_pair < 0 or n_pair == 1:
            for sub_id in range(n_caption):
                sub_list.append([sub_id])
        else:
            sb_ls_ = list(range(n_caption))
            if n_pair > n_caption:
                sb_ls_ = sb_ls_ * (n_pair // n_caption + 1)
                sb_ls_ = sb_ls_[:n_pair]
                for sub_id in np.arange(0, len(sb_ls_), n_pair):
                    sub_list.append(sb_ls_[sub_id: sub_id + n_pair])
            else:
                sb_ls_ = sb_ls_ + sb_ls_[:(
                            ((n_caption + positive_n_pair - 1) // positive_n_pair) * positive_n_pair - n_caption)]
                for sub_id in np.arange(0, len(sb_ls_), positive_n_pair):
                    pos_ls = sb_ls_[sub_id: sub_id + positive_n_pair]
                    sub_list.append(pos_ls)

        for sub_e in sub_list:
            iter2video_pairs_dict[iter_idx_mil_] = (video_id, sub_e)
            iter_idx_mil_ += 1
        iter2video_pairslist_dict[video_id] = sub_list

    video_dict, word_list, start_list, end_list, new_id_file_list, new_id_list = {}, [], [], [], [], []
    csv_files = []

    for feature_idx in iter2video_pairslist_dict:  # sample from each video, has a higher priority than use_mil.
        idx = video_id2idx_dict[feature_idx]
        video_id = howto100_csv['video_id'].values[idx]
        sub_list = iter2video_pairslist_dict[video_id]
        ranint = np.random.randint(0, len(sub_list))
        sub_ids = sub_list[ranint]
        words, starts, ends = _get_text(video_id, n_pair, sub_ids,)
        video, id_file, new_id, start, end, text = _get_video(idx, starts, ends, words)


        video_dict[new_id] = {'start': np.array(start), 'end': np.array(end), 'text': np.array(text)}
        csv_files.append([new_id, id_file])

        # save numpy files to appropriate dir
        with open('./data/howto100m/sample_clips/' + id_file, 'wb') as f:
            np.save(f, video)

    with open('./data/howto100m/HowTo100M_clips.csv', 'w', newline='') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(['video_id', 'feature_file'])
        csvwriter.writerows(csv_files)

    mpu.io.write('./data/howto100m/caption_clips.pickle', video_dict)



