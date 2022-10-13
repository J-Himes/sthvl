import logging
from pathlib import Path

import numpy as np
import torch

from src.helpers import init_helper, data_helper, vsumm_helper, bbox_helper, video_helper
from anchor_based.dsnet import DSNet
import pickle

class DSNET:
    def __init__(self, trained_weights_path='../models/pretrain_ab_basic/checkpoint/tvsum.yml.0.pt'):
        # Creating our model's architecture
        self.model = DSNet('attention', 1024, 128, [4, 8, 16, 32], 8)
        # Setting our model to evaluation mode on our cuda device
        self.model = self.model.eval().to('cuda')

        # Loading model weights
        ckpt_path = trained_weights_path
        state_dict = torch.load(str(ckpt_path), map_location=lambda storage, loc: storage)
        self.model.load_state_dict(state_dict)

    def get_frames_idx(self, video, sampling_rate=1, percent=50):
        # Initialize our video processor with the sample rate
        vp = video_helper.VideoPreprocessor(sampling_rate)

        # Two frames per change point
        seq_len = n_frames = len(video)
        picks = np.arange(0, seq_len) * sampling_rate
        cps = [] # Nx2 array containing start and end change points
        nfps = [] # Nx1 array containing number of frames per change point
        i = 0
        while i < seq_len:
            start = i
            i += 1
            end = i
            i += 1
            if i == seq_len - 1:
                end = i
                i += 1
            cps.append([start, end])
            nfps.append((end - start) + 1)

        with torch.no_grad():
            # Putting input on cuda device
            seq_torch = video.unsqueeze(0).to('cuda')

            # Model making prediction on input
            pred_cls, pred_bboxes = self.model.predict(seq_torch)

            # Clipping predicted bboxes between 0 and seq_len
            pred_bboxes = np.clip(pred_bboxes, 0, seq_len).round().astype(np.int32)

            # Non-maximal suppression to remove overlapping bboxes
            nms_thresh = 0.4
            pred_cls, pred_bboxes = bbox_helper.nms(pred_cls, pred_bboxes, nms_thresh)

            # pred_summ contains True/False for each frame in the video
            proportion = float(percent / 100)
            pred_summ = vsumm_helper.bbox2summary(
                seq_len, pred_cls, pred_bboxes, cps, n_frames, nfps, picks, proportion=proportion)

            # Getting the indices of selected frames
            indices = np.argwhere(pred_summ == True)

            return indices

def main():
    # Creating our model's architecture
    model = DSNet('attention', 1024, 128, [4, 8, 16, 32], 8)
    # Setting our model to evaluation mode on our cuda device
    model = model.eval().to('cuda')

    # Loading model weights
    ckpt_path = '../models/pretrain_ab_basic/checkpoint/tvsum.yml.0.pt'
    state_dict = torch.load(str(ckpt_path), map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    # Loading our video features
    features = pickle.load(open('/home/jhimes/Research/Test3/sthvl/DSNet/datasets/msrvtt/msrvtt_videos_features.pickle', 'rb'))
    feature = torch.tensor(features['video6119'], dtype=torch.float32)
    seq_len = n_frames = len(feature)

    # Initialize our video processor with sample rate of 1
    sample_rate = 1
    vp = video_helper.VideoPreprocessor(sample_rate)

    dsnet_obj = DSNET()
    test = dsnet_obj.get_frames_idx(feature)

    # Approach for evaluate in original DSNet code
    # Adding in our cps code
#    sample_rate = 15
    picks_1 = np.arange(0, seq_len) * sample_rate
    cps_1 = []
    nfps_1 = []
    i = 0

    while i < seq_len:
        start = i * sample_rate
        i += 1
        end = i * sample_rate
        i += 1
        if i == seq_len:
            end = min(n_frames - 1, (i + 1) * sample_rate)
            i += 1

        cps_1.append([start, end])
        nfps_1.append((end - start) + 1)
    cps_1 = np.asarray(cps_1)
    nfps_1 = np.asarray(nfps_1)
    cps = cps_1
    nfps = nfps_1
    print('Test')
    # End of our code

        # WORKING APPROACH
    # # Approach 2, one frame per change point
    # picks = np.arange(0, seq_len) * sample_rate
    # cps = []
    # nfps = []
    # i = 0
    #
    # while i < seq_len:
    #     start = i
    #     i += 1
    #     end = i
    #     i += 1
    #     if i == seq_len - 1:
    #         end = i
    #         i += 1
    #     cps.append([start, end])
    #     nfps.append((end - start) + 1)

    # for i in range(seq_len):
    #     start = i
    #     i += 1
    #     end = i
    #     i += 1
    #     cps.append([start, end])

    indices = None

    print('Predicting summary ...')
    with torch.no_grad():
        # Putting input on cuda device
        seq_torch = feature.unsqueeze(0).to('cuda')

        # Model making prediction on input
        pred_cls, pred_bboxes = model.predict(seq_torch)

        # Clipping predicted bboxes between 0 and seq_len
        pred_bboxes = np.clip(pred_bboxes, 0, seq_len).round().astype(np.int32)

        # Non-maximal suppression to remove overlapping bboxes
        nms_thresh = 0.4
        pred_cls, pred_bboxes = bbox_helper.nms(pred_cls, pred_bboxes, nms_thresh)

        pred_summ = vsumm_helper.bbox2summary(
            seq_len, pred_cls, pred_bboxes, cps, n_frames, nfps, picks, proportion=0.5)

        # Getting the indices of selected frames
        indices = np.argwhere(pred_summ == True)
        print('hello world')
        print(indices)
        print(test)
        print(indices==test)

if __name__ == '__main__':
    main()
