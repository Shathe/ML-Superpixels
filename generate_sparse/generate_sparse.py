import glob
import cv2
import argparse
import random
import os.path
import math




'''
python generate_sparse/generate_sparse.py --dataset ../Datasets/camvid --n_labels 100  --image_format png --default_value 255
This generates the sparse labels from the dataset. the dataset has to have this folder structure:
-dataset
	-images 
		-train
		-test
	-labels
		-train
		-test
Every sparse labeled image will have [n_labels] number of labeled pixels.
The output folder will have the same name as the [dataset]: dataset/sparse_GT/train and dataset/sparse_GT/test
'''
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="Dataset to train", default='../Datasets/camvid')
parser.add_argument("--n_labels", help="Number of pixel labels to have", default=250)
parser.add_argument("--gridlike", help="Whether to have a grid-like srtuctured ground-truth", default=1)
parser.add_argument("--image_format", help="Labeled image format (jpg, jpeg, png...)", default='png')
parser.add_argument("--default_value", help="Value of non-labeled pixels", default=255)
args = parser.parse_args()


path_names = args.dataset.split('/')
if path_names[-1] == '':
	path_names  = path_names[:-1]
dataset_name  = path_names[-1]

sparse_folder = dataset_name + '/sparse_GT'
labels_folder = args.dataset + '/labels'
NUM_LABELS = int(args.n_labels)
grid = bool(int(args.gridlike))

folders = ['test', 'train'] # folders of the dataset

if not os.path.exists(sparse_folder):
    os.makedirs(sparse_folder)

# for every folder (test, train..)
for folder in folders:

	folder_to_write = sparse_folder + '/' + folder
	if not os.path.exists(folder_to_write):
	    os.makedirs(folder_to_write)

	# For every file of the dataset
	for filename in glob.glob(labels_folder + '/' + folder + '/*.' + args.image_format): 
		image_name = filename.split('/')[-1]
		new_filename = folder_to_write + '/' + image_name

		img = cv2.imread(filename, 0)
		sparse = cv2.imread(filename, 0)
		sparse[:,:] = int(args.default_value)

		i_size, j_size = img.shape
		if grid: # Perform grid-like sparse ground-truth
			rate = i_size*1./j_size # rate between height and width
			sqrt = math.sqrt(NUM_LABELS) 
			n_i_points = int(rate*sqrt)+1 # number of lables per coloumn
			n_j_points = int(NUM_LABELS/n_i_points)+1 # number of lables per row
			space_betw_i = int(i_size / n_i_points) # space between every label
			space_betw_j = int(j_size / n_j_points)
			start_i = int((i_size - space_betw_i*(n_i_points-1))/2) # pixel to start labeling
			start_j = int((j_size - space_betw_j*(n_j_points-1))/2)

			for i in xrange (start_i, n_i_points*space_betw_i, space_betw_i):
				for j in xrange (start_j, n_j_points*space_betw_j, space_betw_j ):
						sparse[i,j] = img[i,j] # assign a label 


		else: # Perform random sparse ground-truth		
			for i in xrange (NUM_LABELS):
				i_point = random.randint(0, i_size-1)
				j_point = random.randint(0, j_size-1)
				sparse[i_point,j_point] = img[i_point,j_point]


		cv2.imwrite(new_filename, sparse)
	


print('GENERATION COMPLETED')
