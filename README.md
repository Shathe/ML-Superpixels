# Semantic Segmentation from Sparse Labeling using Multi-Level Superpixels

Implementation of our multi-level superpixel augmentation from sparse labeling presented on the [IROS 2018](https://www.iros2018.org/).

Link to the paper soon.
Link to the slides soon.

## Citing Multi-Level Superpixels 

If you find Multi-Level Superpixels useful in your research, please consider citing:

@inproceedings{alonso2018MLSuperpixel,
  title={Semantic Segmentation from Sparse Labeling using Multi-Level Superpixels},
  author={Alonso, I{\~n}igo and Murillo, Ana C},
  booktitle={IEEE International Conference on Intelligent Robots and Systems},
  year={2018}
}

## Requirements
- Python 2.7
- OpenCV
- Numpy


## Running it all

### Generate a sparse ground-truth from a fully-labeled ground-truth

First of all, if you don't have any sparse labeled images (images with only a few labeled pixels), you have to generate them from a fully-labeled dataset.

This step generates images with some labeled pixels.

The dataset has to have this folder structure:
```
- dataset
	- images 
		- train
		- test
	- labels
		- train
		- test
```
Like the [camvid dataset]( ./Datasets/camvid)


The code that executes this step:
```
python generate_sparse/generate_sparse.py --dataset ./Datasets/camvid --n_labels 500  --gridlike 1 --image_format png --default_value 255
```
Every sparse labeled image will have [n_labels] number of labeled pixels. You can specify if you want the sparse labels to have a grid structure (value 1), or random (value 0) The rest pixels will have the [default_value] value.

The output folder will have the same name as the [dataset]: [dataset]/sparse_GT/train and [dataset]/sparse_GT/test




### Generate superpixels

To generate the superpixels, you have to specify the dataset path as the first argument:
```
sh generate_superpixels/generate_superpixels.sh ./Datasets/camvid/
```
An the superpixels will be generated.
The output folder will have the same name as the [dataset]: [dataset]/superpixels/train and [dataset]/superpixels/test

### Generate augmented ground-truth

To generate the augmented ground-truth, you have to specify the path where the sparse labels and superpixels have been created.
```
python generate_augmented_GT/generate_augmented_GT.py --dataset ./camvid/
```
The output folder will have the same name as the [dataset]: [dataset]/augmented_GT/train and [dataset]/augmented_GT/test



### Evaluation

For the evaluation of the augmented ground-truth, just compare to the original dense labels with the augmented labels. For this, you can execute:
```
python evaluation/evaluate_augmentation.py --labels ../Datasets/camvid/ --generated ./camvid/ --classes 11
```
Specifying the dataset and the generated folder, as well as the number of classes to evaluate.



















