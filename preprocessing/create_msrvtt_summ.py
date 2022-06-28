import pickle
import sys
import random
import numpy as np
from vsumm.vsumm import vsumm
from vsumm.vsumm_skim import vsumm_skim

def main():
    features_path = sys.argv[1]
    summ_type = sys.argv[2]
    percent = int(sys.argv[3])
    output_pathfile = sys.argv[4]

    feature_dict = pickle.load(open(features_path, 'rb'))
    for key in feature_dict:
        indices = list(range(len(feature_dict[key])))
        if summ_type == 'naive':
            frames = len(feature_dict[key])
            count = int(frames * (percent / 100))
            selected_frames = random.sample(indices, count)
        elif summ_type == 'sampling':
            frames = len(feature_dict[key])
            selected_frames = np.arange(frames)
            selected_frames = selected_frames[selected_frames % 2 == 0]
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