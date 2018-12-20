from __future__ import print_function
import os
import numpy as np
import glob
import cv2
import random
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="Dataset to train", default='../camvid/')
parser.add_argument("--n_labels", help="Dataset to train", default=20)
args = parser.parse_args()


labels_files = glob.glob(os.path.join(args.dataset,'augmented_GT','*','*'))
print(len(labels_files))

for folder in ['train', 'test']:
    dir = os.path.join(args.dataset,'labels_colored', folder)
    if not os.path.exists(dir):
        os.makedirs(dir)


label_to_color = {}
r = lambda: random.randint(0, 255)

for label_i in xrange(args.n_labels):
    color_i = tuple([r(), r(), r()])
    label_to_color[str(label_i)] = color_i

print(label_to_color)

for file in labels_files:

    img = cv2.imread(file, 1)
    print (file)
    for label_i in xrange(args.n_labels):
        color_i = label_to_color[str(label_i)] 
        cond = img[:,:,0]==label_i
        img[cond,:]=color_i
    new_file = file.replace('augmented_GT','labels_colored')

    print (new_file)
    cv2.imwrite(new_file, img)


print('Finished')
