import pickle
import numpy as np
from sklearn.cluster import KMeans

# Parameters
# Parameter 1: Feature
# Parameter 2: Sampling rate (every kth frame is chosen)
# Parameter 3: Percentage length of video summary

def vsumm_skim(feature, sampling_rate, percent, skim_length=1.8, fps=1):
	# Choosing number of centers for clustering
	frame_count = len(feature)
	skim_frames_length = fps * skim_length
	num_centroids=int(percent*frame_count/skim_frames_length/100)+1
	kmeans=KMeans(n_clusters=num_centroids).fit(feature)

	centres=[]
	features_transform=kmeans.transform(feature)
	for cluster in range(features_transform.shape[1]):
		centres.append(np.argmin(features_transform.T[cluster]))

	centres=sorted(centres)
	frames_indices=[]
	for centre in centres:
		for idx in range(max(int(centre*sampling_rate-skim_frames_length/2),0),min(int(centre*sampling_rate+skim_frames_length/2)+1,frame_count)):
			frames_indices.append(idx)
	frames_indices=sorted(set(frames_indices))

	return frames_indices


# Example

# feature_file = '../../data/msrvtt/msrvtt_videos_features.pickle'
# videos = pickle.load(open(sys.argv[1], 'rb'))
# first_key = next(iter(videos))
# feature = videos[first_key]
# sampling_rate = 1
# percent = 50
# temp = vsumm_skim(feature, sampling_rate, percent)
# print(temp)