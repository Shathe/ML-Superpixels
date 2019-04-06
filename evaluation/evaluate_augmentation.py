from __future__ import division
import numpy as np
import os.path
import glob
import csv
import sys
import argparse
import math
import cv2
import ntpath
sys.path.append("/Library/Python/2.7/site-packages/numpy-override/")
sys.path.append('/usr/local/lib/python2.7/site-packages')



# Gets the accuracy of the generated images, from the labeled ones
# python accuracy.py --labels labels10  --generated 10labels-ignore-fluor-0.793 --classes 10




# Import arguments, the two folders. Labeled images (points) and genererated
parser = argparse.ArgumentParser()
parser.add_argument('--labels', type=str, required=True)
parser.add_argument('--generated', type=str, required=True)
parser.add_argument('--classes', type=str, required=True)
args = parser.parse_args()

folders = ['test', 'train']

N_CLASES=int(args.classes)

# Counters variables
acertado =np.zeros(N_CLASES)
real = np.zeros(N_CLASES)
predicho = np.zeros(N_CLASES)
DICE_por_clase = np.zeros(N_CLASES)
interseccion_clase = np.zeros(N_CLASES)
union_clase = np.zeros(N_CLASES)
suma_clase = np.zeros(N_CLASES)
matrix = np.zeros((N_CLASES,N_CLASES), np.uint32)
count = 0


for folder in folders:

	# For every image
	for filename in glob.glob(os.path.join(args.labels , 'labels/' , folder) + '/*.*'):
		count = count 
		# Load the generated and labaled image
		name = ntpath.basename(filename)
		points = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
		
		generated = cv2.imread(os.path.join(args.generated , 'augmented_GT/' , folder, name), cv2.IMREAD_GRAYSCALE)

		generated=cv2.resize(generated,(points.shape[1], points.shape[0]), interpolation=cv2.INTER_NEAREST)
		# Resize the generated to the labeled size (labeled pixels can be lost if it is done the other way round)
		# We create an image, which has positives values in the labeled pixels (to accelerate the calculation) only the first time 
		sys.stdout.write('.')
		sys.stdout.flush()
		for i in range(N_CLASES):
			
			acertado[i] = acertado[i] + sum(sum((points == i ) & (generated == i)))
			real[i] =real[i] + sum(sum(points == i))
			predicho[i] =predicho[i] + sum(sum((points >= 0) & (points < N_CLASES) & (generated == i)))

			interseccion_clase[i] = interseccion_clase[i] + sum(sum((generated==i) & (points==i) ))
			union_clase[i] = union_clase[i] + sum(sum(((generated==i) | (points==i)) & (points < N_CLASES) ))
			suma_clase[i] = suma_clase[i] + sum(sum(((generated==i) & (points < N_CLASES) ) )) + sum(sum((points==i)))

			for x in range(N_CLASES):
				matrix[i,x]= matrix[i,x] + sum(sum((points == i) & (generated == x)))
print ('')

for i in range(N_CLASES):
 	print "Class {0}".format(str(i))
 	print "==================="
 	print "Correct {0} out of {1} . Amount of predicted lables of this class {2}".format(str(acertado[i]), str(real[i]), str(predicho[i]))
 	print "Accuracy {0} ".format(str((acertado[i]/real[i])))
print ''
'''
print "Accuracy Coral {0} ".format(str((sum(acertado[0:6])/sum(real[0:6])))) 
print ''
print "Accuracy No Coral {0} ".format(str((sum(acertado[6:10])/sum(real[6:10])))) 
print ''
'''

media = 0.00
media2 = 0.00
real_classes = N_CLASES
for i in range(N_CLASES):
        for j in range(N_CLASES):
		if i == j:
			if real[i] > 0:
				media = media + float(matrix[i,j]/real[i])
				if 	predicho[i] != 0:
					media2 = media2 + float(matrix[i,j]/predicho[i])

			else:
				real_classes-=1
		#print "%0.2f	" % float(matrix[i,j]/real[i]),




print "Recall (per pixel) {0} ".format(str((sum(acertado)/sum(real)))) 
print ''

print "Recall (mean per class) {0} ".format(media/real_classes)
print ''

print "Accuracy (per pixel) {0} ".format(str((sum(acertado)/sum(predicho)))) 
print ''

print "Accuracy (mean per class) {0} ".format(media2/real_classes)
print ''

IoU =  np.array(interseccion_clase)/np.array(union_clase)
where_are_NaNs = np.isnan(IoU)
IoU[where_are_NaNs] = 0
IoU=IoU[IoU != 0]

print('IoU (mean per class): '  + str(np.mean(IoU)))
IoU =  np.array(sum(interseccion_clase))/np.array(sum(union_clase))
print('IoU (per pixel): '  + str(IoU))

exit;
