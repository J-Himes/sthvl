import pickle
from tqdm import tqdm

def main():
    train_pkl = '../data/Charades_v1_features_rgb_1024_trim_32_32/train.pkl'
    val_pkl = '../data/Charades_v1_features_rgb_1024_trim_32_32/test.pkl'

    with open(train_pkl, 'rb') as f1:
        train = pickle.load(f1)
        with open(val_pkl, 'rb') as f2:
            test = pickle.load(f2)

            common_keys = []
            for t1 in tqdm(train):
                for t2 in test:
                    if t1[-10:] == t2[-10:]:
                        common_keys.append(t1[-10:])

            print(common_keys)
            print(len(common_keys))

if __name__ == "__main__":
    main()