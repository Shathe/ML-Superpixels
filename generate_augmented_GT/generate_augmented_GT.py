import glob
import cv2
import numpy as np
import csv
from collections import Counter
import argparse
import os
import math

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="Dataset to train", default='./Datasets/camvid')
parser.add_argument("--image_format", help="Labeled image format (jpg, jpeg, png...)", default='png')
parser.add_argument("--default_value", help="Value of non-labeled pixels", default=255)
parser.add_argument("--number_levels", help="Number of max iterations", default=15)
parser.add_argument("--start_n_superpixels", help="Number of superpixels in the first iteration", default=3000)
parser.add_argument("--last_n_superpixels", help="Number of superpixels in the last iteration", default=30)
args = parser.parse_args()



DEFAULT_VALUE=int(args.default_value)



NL=int(args.number_levels)
start_superpixels=int(args.start_n_superpixels)
last_superpixels=int(args.last_n_superpixels)
csv_sizes =[] 
reduction_factor = math.pow(float(last_superpixels)/start_superpixels, 1./(NL-1))
for level in xrange(NL):
	csv_sizes =  csv_sizes + [int(round(start_superpixels * math.pow(reduction_factor, level)))]

path_names = args.dataset.split('/')
if path_names[-1] == '':
	path_names  = path_names[:-1]
directorio  = path_names[-1]

sparse_dir = os.path.join( directorio ,'sparse_GT')
out_dir = os.path.join(directorio , 'augmented_GT')
superpixels_dir= os.path.join(directorio, 'superpixels')

folders = ['test', 'train']

# Execute superpixel genration
size_sup_string = " "
for size in csv_sizes:
	size_sup_string = size_sup_string + str(size) + " "

# Generate superpixels
os.system("sh generate_superpixels/generate_superpixels.sh " + args.dataset + size_sup_string)  


class Superpixel:
	def __init__(self):
		self.lista_x = np.array([])
		self.lista_y = np.array([])



#Given a superpixel and a GT image, returns the label value of the superpixel
def label_mayoria_x_y(superpixel, gt):
	#pixel label values of the superpixels
	pixel_values = gt[superpixel.lista_x.astype(int), superpixel.lista_y.astype(int)]
	#pixel label values of the superpixels excluding the default value
	values_labels = pixel_values[pixel_values<DEFAULT_VALUE]
	#Returns the value which appears the most
	if len(values_labels) == 0:
		return DEFAULT_VALUE
	else:
		count = Counter(values_labels)
		return count.most_common()[0][0]








#Given a csv file with segmentations (csv_name) and a sparse GT image (gt_name), returns Superpixel-augmented GT image
def image_superpixels_gt(csv_name, gt_name ):
	print(gt_name)
	gt = cv2.imread(gt_name,0)
	blank_image = np.zeros((gt.shape[0], gt.shape[1],1), np.uint8)

	i=0
	superpixels={}

	#For each csv segmentation file, creates the Superpixel class
	with open(csv_name, 'rb') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
		for row in spamreader:
			fila = row[0].split(',')
			fila_count = len(fila) 
			
			for j in xrange(fila_count):
				superpixel_index = fila[j] 
				#The pixel here is (i, j). (superpixel_index) is the segmentation which the pixel belongs to
				if superpixel_index not in superpixels.keys():
					superpixels[superpixel_index]=Superpixel()
				#Add the pixel  to the Superpixel instance
				superpixels[superpixel_index].lista_x=np.append(superpixels[superpixel_index].lista_x,i)
				superpixels[superpixel_index].lista_y=np.append(superpixels[superpixel_index].lista_y,j)
			i = i + 1

		
	#For each superpixel, gets its label value and writes it into the image to return
	for index in xrange(len(superpixels)):
		label_superpixel = label_mayoria_x_y(superpixels[str(index)], gt)
		blank_image[superpixels[str(index)].lista_x.astype(int), superpixels[str(index)].lista_y.astype(int)] = int(label_superpixel)

	return blank_image



def generar_augmentedGT():

	for folder in folders:

		in_folder = os.path.join(sparse_dir , folder)
		out_folder = os.path.join(out_dir , folder)
		superpixels_folder = os.path.join(superpixels_dir , folder)

		if not os.path.exists(out_folder):
			os.makedirs(out_folder)

		for filename in glob.glob(in_folder + '/*.' + args.image_format): #imagenes test a crear patches
			gt_name = filename.split('/')[-1]
			gt_filename = os.path.join( out_folder ,  gt_name)			
			
			#For each different segmentation generated
			for index in xrange(len(csv_sizes)):
				print(csv_sizes[index])
				csv_name = os.path.join(superpixels_folder, 'superpixels_'+ str(csv_sizes[index]) ,  gt_name.replace('.' + args.image_format,'') + '.csv')

				if index == 0:
					#creates the first one (it has to be the more detailed one, the segmentation with more segments)
					image_gt_new = image_superpixels_gt(csv_name, filename )
				else:
					#Mask it with the less detailed segmentations in order to fill the areas with no valid labels
					image_gt_new_low = image_superpixels_gt(csv_name, filename )
					image_gt_new[image_gt_new==DEFAULT_VALUE]=image_gt_new_low[image_gt_new==DEFAULT_VALUE]
				#cv2.imwrite(gt_filename.replace(gt_name, gt_name+'_'+str(csv_sizes[index])),image_gt_new)


			# out_dir
			cv2.imwrite(gt_filename,image_gt_new)
			



generar_augmentedGT()
print('GENERATION COMPLETED')







