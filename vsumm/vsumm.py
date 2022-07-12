# k means clustering to generate video summary
import numpy as np
from sklearn.cluster import KMeans

def vsumm(feature, sampling_rate, percent,):
	# converting percentage to actual number
	num_centroids=int(percent*len(feature)/100)
	if (len(feature)/sampling_rate) < num_centroids:
		print("Samples too less to generate such a large summary")
		print("Changing to maximum possible centroids")
		num_centroids=len(feature)/sampling_rate

	kmeans=KMeans(n_clusters=num_centroids).fit(feature)
	summary_frames=[]

	# transforms into cluster-distance space (n_cluster dimensional)
	hist_transform=kmeans.transform(feature)
	frame_indices=[]
	for cluster in range(hist_transform.shape[1]):
		frame_indices.append(np.argmin(hist_transform.T[cluster]))
	
	# frames generated in sequence from original video
	frame_indices=sorted(frame_indices)
	# summary_frames=[feature[i] for i in frame_indices]

	return frame_indices

# feature_file = '../../data/msrvtt/msrvtt_videos_features.pickle'
# videos = pickle.load(open(sys.argv[1], 'rb'))
# first_key = next(iter(videos))
# feature = videos[first_key]
# sampling_rate = 1
# percent = 50
# temp = vsumm(feature, sampling_rate, percent)
# print(temp)