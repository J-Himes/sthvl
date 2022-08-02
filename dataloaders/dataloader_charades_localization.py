from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
from vsumm.vsumm import vsumm
from vsumm.vsumm_skim import vsumm_skim
import numpy as np
import csv
import pandas as pd
from collections import defaultdict
import json
import random
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


class Charades_Localization_DataLoader(Dataset):
    """Charades train dataset loader."""
    def __init__(
            self,
            csv_path,
            json_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            split_type="",
            summ_type=None
    ):
        self.csv = pd.read_csv(csv_path)
        self.data = json.load(open(json_path, 'r'))
        path_cmd = features_path + '/*'
        self.feature_dict = glob(path_cmd, recursive = False)
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.summ_type=summ_type

        self.feature_size = 1024  # Hardcoded, same size as MSRVTT

        assert split_type in ["train", "val", "test"]
        video_ids = [self.feature_dict[idx][-5:] for idx in range(len(self.feature_dict))]
        self.feature_dict = dict(zip(video_ids, self.feature_dict))
        choiced_video_ids = []
        for key, value in self.data.items():
            if value['subset'][:len(split_type)] == split_type:
                choiced_video_ids.append(key)

        self.labels, pop_list = make_dataset(self.data, choiced_video_ids, self.feature_dict, num_classes=157)
        for key in pop_list:
            choiced_video_ids.pop(key[0])

        self.sample_len = 0
        self.sentences_dict = {}
        self.video_sentences_dict = defaultdict(list)
        i = 0
        for key in choiced_video_ids:
            csv_value = self.csv.loc[self.csv['id'] == key]
            try:
                description = csv_value['descriptions'].values[0]
            except:
                continue
            self.sentences_dict[i] = (key, description)
            self.video_sentences_dict[key].append(description)
            i += 1

        self.sample_len = len(self.sentences_dict)

    def __len__(self):
        return self.sample_len

    def _get_text(self, video_id, caption=None):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)
        pairs_masked_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_token_labels = np.zeros((k, self.max_words), dtype=np.long)

        pairs_input_caption_ids = np.zeros((k, self.max_words), dtype=np.long)
        pairs_output_caption_ids = np.zeros((k, self.max_words), dtype=np.long)
        pairs_decoder_mask = np.zeros((k, self.max_words), dtype=np.long)

        for i, video_id in enumerate(choice_video_ids):
            words = []
            words = ["[CLS]"] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + ["[SEP]"]

            # Mask Language Model <-----
            token_labels = []
            masked_tokens = words.copy()
            for token_id, token in enumerate(masked_tokens):
                if token_id == 0 or token_id == len(masked_tokens) - 1:
                    token_labels.append(-1)
                    continue
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.15:
                    prob /= 0.15
                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        masked_tokens[token_id] = "[MASK]"
                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        masked_tokens[token_id] = random.choice(list(self.tokenizer.vocab.items()))[0]
                    # -> rest 10% randomly keep current token
                    # append current token to output (we will predict these later)
                    try:
                        token_labels.append(self.tokenizer.vocab[token])
                    except KeyError:
                        # For unknown words (should not occur with BPE vocab)
                        token_labels.append(self.tokenizer.vocab["[UNK]"])
                        # print("Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
                else:
                    # no masking token (will be ignored by loss function later)
                    token_labels.append(-1)
            # -----> Mask Language Model

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            masked_token_ids = self.tokenizer.convert_tokens_to_ids(masked_tokens)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                masked_token_ids.append(0)
                token_labels.append(-1)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words
            assert len(masked_token_ids) == self.max_words
            assert len(token_labels) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)
            pairs_masked_text[i] = np.array(masked_token_ids)
            pairs_token_labels[i] = np.array(token_labels)

            # For generate captions
            if caption is not None:
                caption_words = self.tokenizer.tokenize(caption)
            else:
                caption_words = self._get_single_text(video_id)
            if len(caption_words) > total_length_with_CLS:
                caption_words = caption_words[:total_length_with_CLS]
            input_caption_words = ["[CLS]"] + caption_words
            output_caption_words = caption_words + ["[SEP]"]

            # For generate captions
            input_caption_ids = self.tokenizer.convert_tokens_to_ids(input_caption_words)
            output_caption_ids = self.tokenizer.convert_tokens_to_ids(output_caption_words)
            decoder_mask = [1] * len(input_caption_ids)
            while len(input_caption_ids) < self.max_words:
                input_caption_ids.append(0)
                output_caption_ids.append(0)
                decoder_mask.append(0)
            assert len(input_caption_ids) == self.max_words
            assert len(output_caption_ids) == self.max_words
            assert len(decoder_mask) == self.max_words

            pairs_input_caption_ids[i] = np.array(input_caption_ids)
            pairs_output_caption_ids[i] = np.array(output_caption_ids)
            pairs_decoder_mask[i] = np.array(decoder_mask)

        return pairs_text, pairs_mask, pairs_segment, pairs_masked_text, pairs_token_labels, \
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, choice_video_ids

    def _get_single_text(self, video_id):
        rind = random.randint(0, len(self.sentences[video_id]) - 1)
        caption = self.sentences[video_id][rind]
        words = self.tokenizer.tokenize(caption)
        return words

    def _get_video(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)

        video = np.zeros((len(choice_video_ids), self.max_frames, self.feature_size), dtype=np.float)
        for i, video_id in enumerate(choice_video_ids):
            load_path = self.feature_dict[video_id] + '/' + self.feature_dict[video_id][-5:] + '.npy'
            video_slice = np.load(load_path)

            percent = 50
            indices = list(range(len(video_slice)))
            if self.summ_type == 'sampling':
                frames = len(video_slice)
                selected_frames = np.arange(frames)
                if percent != 75:
                    rate = percent / 100
                    selected_frames = selected_frames[
                        selected_frames * rate == (selected_frames * rate).astype(int)]
                else:
                    rate = 0.25
                    selected_frames = selected_frames[
                        selected_frames * rate == (selected_frames * rate).astype(int)]
                    selected_frames = np.delete(indices, selected_frames)
            elif self.summ_type == 'vsumm_key':
                selected_frames = vsumm(video_slice, 1, percent)
            elif self.summ_type == 'vsumm_skim':
                selected_frames = vsumm_skim(video_slice, 1, percent)
            if self.summ_type != 'None':
                deleted_frames = [x for x in indices if x not in selected_frames]
                video_slice = np.delete(video_slice, deleted_frames, axis=0)

            if self.max_frames < video_slice.shape[0]:
                video_slice = video_slice[:self.max_frames]

            slice_shape = video_slice.shape
            max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_shape[0] else slice_shape[0]
            if len(video_slice) < 1:
                print("video_id: {}".format(video_id))
            else:
                video[i][:slice_shape[0]] = video_slice

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        # Mask Frame Model <-----
        video_labels_index = [[] for _ in range(len(choice_video_ids))]
        masked_video = video.copy()
        for i, video_pair_ in enumerate(masked_video):
            for j, _ in enumerate(video_pair_):
                if j < max_video_length[i]:
                    prob = random.random()
                    # mask token with 15% probability
                    if prob < 0.15:
                        masked_video[i][j] = [0.] * video.shape[-1]
                        video_labels_index[i].append(j)
                    else:
                        video_labels_index[i].append(-1)
                else:
                    video_labels_index[i].append(-1)
        video_labels_index = np.array(video_labels_index, dtype=np.long)
        # -----> Mask Frame Model

        return video, video_mask, masked_video, video_labels_index

    def __getitem__(self, idx):
        video_id, caption = self.sentences_dict[idx]
        label = self.labels[idx]

        pairs_text, pairs_mask, pairs_segment, \
        pairs_masked_text, pairs_token_labels, \
        pairs_input_caption_ids, pairs_decoder_mask, \
        pairs_output_caption_ids, choice_video_ids = self._get_text(video_id, caption)

        video, video_mask, masked_video, video_labels_index = self._get_video(choice_video_ids)
        label = label[1][:, :video.shape[1]]

        return pairs_text, pairs_mask, pairs_segment, video, video_mask, \
               pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, label
