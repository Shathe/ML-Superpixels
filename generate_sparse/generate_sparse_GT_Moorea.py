import glob
import cv2
import numpy as np
import csv
import argparse
import ntpath
import PIL
from PIL import Image

import os.path

import random



#Name of the classes which appear in the csv file
name_to_number={}
'''
5 CORAL CLASSES
4 NO CORAL CLASSES

Acropora, Pocillopora, Porites, Pavona, Montipora, 
Macro, CCA, Turf, Sand. 
'''
name_to_number['Acrop'] = 0
name_to_number['ACROP'] = 0
name_to_number['Pocill'] = 1
name_to_number['POCILL'] = 1
name_to_number['Porit'] = 2 
name_to_number['PORIT'] = 2 
name_to_number['PAVON'] = 3
name_to_number['Pavon'] = 3
name_to_number['MONTI'] = 4
name_to_number['Monti'] = 4
name_to_number['Macro'] = 5
name_to_number['MACRO'] = 5
name_to_number['CCA'] = 6 
name_to_number['Turf'] = 7 
name_to_number['TURF'] = 7 
name_to_number['Sand'] = 8 
name_to_number['SAND'] = 8 

borde_x_mayor=3000
borde_x_menor=3000
borde_y_mayor=3000
borde_y_menor=3000
 



train_file = 'Moorea/train.txt'


try:
    os.remove(train_file)

except OSError:
    pass

f_train = open(train_file, 'w')


sumacuidado=0


#748 and 1123 are the Height and Width of the Gt images to create.
# Raw images are 5 times bigger, that's why in the code 'row[' row'])/5' it's done

for filename in glob.glob('Moorea/2008/*.txt'): #imagenes test a crear patches
	print(filename)

	img = cv2.imread(filename.replace('.txt',''))

	img=cv2.resize(img,(img.shape[1]/4, img.shape[0]/4))

	height, width = img.shape[:2]
	new_image = np.zeros((height, width , 1), np.uint8)
	new_image[:,:]=255
	with open(filename) as f:
		content = f.readlines()
		# you may also want to remove whitespace characters like `\n` at the end of each line
		content = [x.strip() for x in content] 
		valid_labels = 0
		for linea in content:
			if '#' not in linea:

				linea = linea.split(';')
				row = int(str(linea[0]).replace(' ',''))/4
				col = int(str(linea[1]).replace(' ',''))/4
				label = str(linea[2]).replace(' ','')
				if label in name_to_number:
					number_label = int(name_to_number[label])
					new_image[row,col]=number_label
					valid_labels = valid_labels + 1
	print(valid_labels)
	if valid_labels > 150:
		sumacuidado = sumacuidado +sum(sum((new_image<255) & (new_image>8)))
		print('Moorea/sparse_GT/'+str(ntpath.basename(filename).replace('.txt',''))+'.png')
		print(sum(sum(new_image < 255)))
		cv2.imwrite('Moorea/sparse_GT/'+str(ntpath.basename(filename).replace('.txt','').replace('.jpg','')+'.png'),new_image)
		cv2.imwrite('Moorea/images/'+str(ntpath.basename(filename).replace('.txt','').replace('.jpg','')+'.png'),img)
		f_train.write(ntpath.basename(filename).replace('.txt','').replace('.jpg','') + '\n')  # python will convert \n to os.linesep
		print(sum(sum((new_image<255) & (new_image>8))))
		gt = cv2.imread('Moorea/sparse_GT/'+str(ntpath.basename(filename).replace('.txt','').replace('.jpg','')+'.png'),0) 
		print(sum(sum((gt<255) & (gt>8))))
 

for filename in glob.glob('Moorea/2009/*.txt'): #imagenes test a crear patches
	print(filename)
	img = cv2.imread(filename.replace('.txt',''))
	img=cv2.resize(img,(img.shape[1]/4, img.shape[0]/4))
	height, width = img.shape[:2]
	new_image = np.zeros((height, width , 1), np.uint8)
	new_image[:,:]=255
	with open(filename) as f:
		content = f.readlines()
		# you may also want to remove whitespace characters like `\n` at the end of each line
		content = [x.strip() for x in content] 
		valid_labels = 0
		for linea in content:
			if '#' not in linea:

				linea = linea.split(';')
				row = int(str(linea[0]).replace(' ',''))/4
				col = int(str(linea[1]).replace(' ',''))/4
				label = str(linea[2]).replace(' ','')
				if label in name_to_number:
					number_label = int(name_to_number[label])
					new_image[row,col]=number_label
					valid_labels = valid_labels + 1
	if valid_labels > 150:
		sumacuidado = sumacuidado +sum(sum((new_image<255) & (new_image>8)))
		cv2.imwrite('Moorea/sparse_GT/'+str(ntpath.basename(filename).replace('.txt','').replace('.jpg','')+'.png'),new_image)
		cv2.imwrite('Moorea/images/'+str(ntpath.basename(filename).replace('.txt','').replace('.jpg','')+'.png'),img)
		f_train.write(ntpath.basename(filename).replace('.txt','').replace('.jpg','') + '\n')  # python will convert \n to os.linesep



for filename in glob.glob('Moorea/2010/*.txt'): #imagenes test a crear patches
	print(filename)
	img = cv2.imread(filename.replace('.txt',''))
	img=cv2.resize(img,(img.shape[1]/4, img.shape[0]/4))
	height, width = img.shape[:2]
	new_image = np.zeros((height, width , 1), np.uint8)
	new_image[:,:]=255
	with open(filename) as f:
		content = f.readlines()
		# you may also want to remove whitespace characters like `\n` at the end of each line
		content = [x.strip() for x in content] 
		valid_labels = 0
		for linea in content:
			if '#' not in linea:

				linea = linea.split(';')
				row = int(str(linea[0]).replace(' ',''))/4
				col = int(str(linea[1]).replace(' ',''))/4
				label = str(linea[2]).replace(' ','')
				if label in name_to_number:
					number_label = int(name_to_number[label])
					new_image[row,col]=number_label
					valid_labels = valid_labels + 1
	if valid_labels > 150:
		sumacuidado = sumacuidado +sum(sum((new_image<255) & (new_image>8)))
		cv2.imwrite('Moorea/sparse_GT/'+str(ntpath.basename(filename).replace('.txt','').replace('.jpg','')+'.png'),new_image)
		cv2.imwrite('Moorea/images/'+str(ntpath.basename(filename).replace('.txt','').replace('.jpg','')+'.png'),img)
		f_train.write(ntpath.basename(filename).replace('.txt','').replace('.jpg','') + '\n')  # python will convert \n to os.linesep




print(sumacuidado)


f_train.close()





'''

new_image = np.zeros((748, 1123,1), np.uint8)
new_image[:,:]=255
filename=''
with open('Dataset/annotations.csv') as csvfile:
	reader = csv.DictReader(csvfile)
	#read annotations until the filename changes
	for row in reader:

		if filename!=row['filename'] and filename!='':
			#If the filename changes, write the new image
			cv2.imwrite('Moorea/sparse_GT/'+str(filename)+'.png',new_image)

		#Read annotation
		filename=row['filename']
		x = int(int(row[' row'])/5)
		y = int(int(row[' columns'])/5)
		label = row[' label']
		number_label = int(name_to_number[label])
		new_image[x,y]=number_label

#write last image
cv2.imwrite('Moorea/sparse_GT/'+str(filename)+'.png',new_image)
'''

