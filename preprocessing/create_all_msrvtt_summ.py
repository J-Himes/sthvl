import pickle
import sys
import os
import random
import numpy as np
from vsumm.vsumm import vsumm
from vsumm.vsumm_skim import vsumm_skim
from tqdm import tqdm

def main():
    features_path = sys.argv[1]
    output_pathfile_template = '../data/msrvtt/summ/msrvtt_videos_features'

    summ_types = ['naive', 'vsumm_key', 'vsumm_skim']
    percents = [10, 25, 50, 75]
    for summ_type in tqdm(summ_types):
        print(summ_type)
        for percent in tqdm(percents):
            print(percent)
            output_pathfile = output_pathfile_template + '_' + summ_type + '_' + str(percent) + '.pickle'
            if os.path.exists(output_pathfile):
                continue

            feature_dict = pickle.load(open(features_path, 'rb'))
            for key in feature_dict:
                indices = list(range(len(feature_dict[key])))
                if summ_type == 'naive':
                    frames = len(feature_dict[key])
                    count = int(frames * (percent / 100))
                    selected_frames = random.sample(indices, count)
                elif summ_type == 'vsumm_key':
                    selected_frames = vsumm(feature_dict[key], 1, percent)
                elif summ_type == 'vsumm_skim':
                    selected_frames = vsumm_skim(feature_dict[key], 1, percent)
                deleted_frames = [x for x in indices if x not in selected_frames]
                feature_dict[key] = np.delete(feature_dict[key], deleted_frames, axis=0)

            with open(output_pathfile, 'wb') as f:
                pickle.dump(feature_dict, f)

if __name__ == "__main__":
    main()