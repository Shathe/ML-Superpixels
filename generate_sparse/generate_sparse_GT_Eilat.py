import glob
import cv2
import numpy as np
import csv
import argparse


#Name of the classes which appear in the csv file
name_to_number={}
'''
1: Faviidae
2: Stylophora
3: Platygyra
4: Acropora
5: Pocillopora
6: Other Hard Coral
7: Bare-subst.
8: Millepora
9: Unknown
10: Other Inv.
'''

name_to_number[' Faviidae'] = 0
name_to_number[' Stylophora'] = 1
name_to_number[' Platygyra'] = 2
name_to_number[' Acropora'] = 3
name_to_number[' Pocillopora'] = 4
name_to_number[' Other Hard Coral'] = 5
name_to_number[' Bare-subst.'] = 6 
name_to_number[' Millepora'] = 7 
name_to_number[' Unknown'] = 8 
name_to_number[' Other Inv.'] = 9 

label_count={}
#748 and 1123 are the Height and Width of the Gt images to create.
# Raw images are 5 times bigger, that's why in the code 'row[' row'])/5' it's done
new_image = np.zeros((748, 1123,1), np.uint8)
new_image[:,:]=255
filename=''
with open('Dataset/annotations.csv') as csvfile:
	reader = csv.DictReader(csvfile)
	#read annotations until the filename changes
	for row in reader:

		if filename!=row['filename'] and filename!='':
			#If the filename changes, write the new image
			cv2.imwrite('sparse_GT/'+str(filename)+'.png',new_image)

		#Read annotation
		filename=row['filename']
		x = int(int(row[' row'])/5)
		y = int(int(row[' columns'])/5)
		label = row[' label']
		number_label = int(name_to_number[label])
		new_image[x,y]=number_label

#write last image
cv2.imwrite('sparse_GT/'+str(filename)+'.png',new_image)


