import glob
import cv2
import numpy as np
import csv
from collections import Counter
import argparse
import os.path
 
# Edges of annotations of the sparse GT images
x_menor=159
x_mayor=591
y_menor=220
y_mayor=906

# Pixel margin to rely on annotations away from the edges
margen_pixeles=45
default_value=255
height=748
width=1123

# margins_label_value = 6 # The default value should be label 6 (sand) or 8 (unkown). It's bad to have ignore label values in the same areas (edges/margins)
all_ignore_to_margin_value = False
csv_sizes =[1500,1200,960,760,610,490,390,310,250,200,160,120,90,60,40,20,7]
images_files = ['test', 'train']
Last_label_coral=5
method='Eilat/fluor/SEEDS/'
class Superpixel:
	def __init__(self):

		self.lista_x = np.array([])
		self.lista_y = np.array([])



#Given a superpixel and a GT image, returns the label value of the superpixel
def label_mayoria_x_y(superpixel, gt):
	#pixel label values of the superpixels

	pixel_values = gt[superpixel.lista_x.astype(int), superpixel.lista_y.astype(int)]
	#pixel label values of the superpixels excluding the default value
	values_labels = pixel_values[pixel_values<default_value]

	#Returns the value which appears the most
	if len(values_labels) == 0:
		return default_value
	else:
		count = Counter(values_labels)
		return count.most_common()[0][0]








#Given a csv file with segmentations (csv_name) and a sparse GT image (gt_name), returns Superpixel-augmented GT image in the (filename) path
def image_superpixels_gt(filename, csv_name, gt_name ):
	gt = cv2.imread(gt_name,0)
	blank_image = np.zeros((height, width,1), np.uint8)
	blank_image[:,:] = default_value
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

			#columnas cuadno termina

	#For each superpixel, gets its label value and writes it into the image to return
	for index in xrange(len(superpixels)):
		label_superpixel = label_mayoria_x_y(superpixels[str(index)], gt)
		blank_image[superpixels[str(index)].lista_x.astype(int), superpixels[str(index)].lista_y.astype(int)] = int(label_superpixel)

	return blank_image



def generar_augmentedGT(readfile):

	with open(readfile + '.txt', 'r') as f:
		for filenames in f:
			
			filename=filenames.splitlines()[0]
			gt_name = 'Eilat/sparse_GT/' + filename + '.png'
			#For each different segmentation generated
			for index in xrange(len(csv_sizes)):
				if index == 0:
					#creates the first one (it has to be the more detailed one, the segmentation with more segments)
					csv_name= method + readfile + '/superpixels_'+ str(csv_sizes[index]) + '/' + filename + '.csv'
					print(csv_name)
					print(os.path.isfile(csv_name))
					image_gt_new = image_superpixels_gt(filename, csv_name, gt_name )

				else:
					#Mask it with the less detailed segmentations in order to fill the areas with no valid labels
					csv_name= method + readfile + '/superpixels_'+ str(csv_sizes[index]) + '/' + filename + '.csv'
					image_gt_new_low = image_superpixels_gt(filename, csv_name, gt_name )
					image_gt_new[image_gt_new==default_value]=image_gt_new_low[image_gt_new==default_value]

				#cv2.imwrite(method+'GT_SUPERPIXELS/10labels/'+ readfile + '/' + filename +'-'+str(csv_sizes[index]) +'.png',image_gt_new)
				#print(method+'GT_SUPERPIXELS/10labels/'+ readfile + '/' + filename +'-'+str(csv_sizes[index]) +'.png')

			#Areas with no information have to be with the ignore label value
			image_gt_new[:x_menor-margen_pixeles,:]=default_value #margins_label_value
			image_gt_new[x_mayor+margen_pixeles:,:]=default_value #margins_label_value
			image_gt_new[:,:y_menor-margen_pixeles]=default_value #margins_label_value
			image_gt_new[:,y_mayor+margen_pixeles:]=default_value #margins_label_value

			if all_ignore_to_margin_value:
					image_gt_new[image_gt_new==default_value]= margins_label_value
			print(method+'GT_SUPERPIXELS/10labels/'+ readfile + '/' + filename + '.png')
			cv2.imwrite(method+'GT_SUPERPIXELS/10labels/'+ readfile + '/' + filename + '.png',image_gt_new)

			#Generats also the binary coral vs no coral
			image_gt_new[image_gt_new<=Last_label_coral]=1 #coral
			image_gt_new[(image_gt_new < default_value) & (image_gt_new > Last_label_coral)]=0#no_coral

			cv2.imwrite(method+'GT_SUPERPIXELS/2labels/'+ readfile + '/' + filename + '.png',image_gt_new)





for filename in images_files:
	generar_augmentedGT(filename)















