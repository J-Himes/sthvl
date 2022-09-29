import pickle
from tqdm import tqdm

def main():
    pkls = ['../data/Charades_v1_features_rgb_1024_trim_32_32/train.pkl', '../data/Charades_v1_features_rgb_1024_trim_32_32/test.pkl']
    save_paths = ['../data/Charades_v1_features_rgb_1024_trim_32_32/train_trimmed.pkl', '../data/Charades_v1_features_rgb_1024_trim_32_32/test_trimmed.pkl']

    all_paths = []
    for i2, pkl in enumerate(pkls):
        with open(pkl, 'rb') as f:
            paths = pickle.load(f)
            all_paths.append(paths)

            to_remove = []
            key = ''
            prev_key = ''
            prev_path = ''
            for i, path in tqdm(enumerate(paths)):
                key = path[-10:-5]
                if key != prev_key and i != 0:
                    to_remove.append(prev_path)
                prev_path = path
                prev_key = key

            new_paths = [x for x in paths if x not in to_remove]
            with open(save_paths[i2], 'wb') as f2:
                pickle.dump(new_paths, f2)

    for i2, pkl in enumerate(save_paths):
        with open(pkl, 'rb') as f:
            paths = pickle.load(f)
            all_paths.append(paths)

    print()


if __name__ == "__main__":
    main()