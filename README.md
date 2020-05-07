# Semantic Segmentation from Sparse Labeling using Multi-Level Superpixels

Implementation of our multi-level superpixel augmentation from sparse labeling presented on the [IROS 2018](https://www.iros2018.org/).

Regarding our next work, *CoralSeg: Learning Coral Segmentation from Sparse Annotations*, we improve the algorithm regarding robustness and efficiency on memory (now it supports high resolution images) and speed (now it goes between x2 and x3 faster).
 

## Citing Multi-Level Superpixels 

If you find Multi-Level Superpixels useful in your research, please consider citing:
```
@inproceedings{alonso2019CoralSeg,
  title={CoralSeg: Learning Coral Segmentation from Sparse Annotations},
  author={Alonso, I{\~n}igo and Yuval, Matan and Eyal, Gal and Treibitz, Tali and Murillo, Ana C},
  booktitle={Journal of Field Robotics},
  year={2019}
}

@inproceedings{alonso2018MLSuperpixel,
  title={Semantic Segmentation from Sparse Labeling using Multi-Level Superpixels},
  author={Alonso, I{\~n}igo and Murillo, Ana C},
  booktitle={IEEE International Conference on Intelligent Robots and Systems (IROS)},
  year={2018}
}
```

## Requirements
- Python 2.7
- OpenCV
- Numpy


## Running it all

First of all please, go to [this repository](https://github.com/davidstutz/superpixels-revisited), clone it, [install and compile it](https://github.com/davidstutz/superpixels-revisited#building) and replace the generated binaries in [this folder](https://github.com/Shathe/ML-Superpixels/tree/master/bin) (delete the folders inside ./bin of this repo and copy the folders generated in the superpixels-revisited repo).

The current compiled files may not work for your computer.

For windows users, thanks to [JamesPatrick1014](https://github.com/JamesPatrick1014) compiled and provided the .exe files. 
For using the, just change the first line of the .sh file so these are called instead.

### Generate a sparse ground-truth from a dense-labeled ground-truth

If your segmentation ground truth labels are dense, you can still simulate a sparse one, generating the sparse labeled images (images with only a few labeled pixels) with the following step.

The data should have this folder structure:
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


To run this:
```
python generate_sparse/generate_sparse.py --dataset ./Datasets/camvid --n_labels 500  --gridlike 1 --image_format png --default_value 255
```
Every sparse labeled image will have [n_labels] number of labeled pixels. You can specify if you want the sparse labels to have a grid structure (value 1), or random (value 0) The rest of the pixels will have the [default_value] value.

The output folder will have the same name as the [dataset]: [dataset]/sparse_GT/train and [dataset]/sparse_GT/test



### Generate augmented ground-truth

To generate the augmented ground-truth, you have to specify the path where the sparse labels and superpixels have been created.
```
python generate_augmented_GT/generate_augmented_GT.py --dataset ./Datasets/camvid --number_levels 15 --start_n_superpixels 3000 --last_n_superpixels 30
```
The output folder will have the same name as the [dataset]: [dataset]/augmented_GT/train and [dataset]/augmented_GT/test



### Evaluation

For the evaluation of the quality of the augmented ground-truth, we compare it to the original dense labels. For this, you can run the following (specifying the dataset and the generated folder, as well as the number of classes to evaluate):
```
python evaluation/evaluate_augmentation.py --labels ./Datasets/camvid/ --generated ./camvid/ --classes 11
```




## Training a semantic segmentation model
[This other repository](https://github.com/Shathe/Semantic-Segmentation-Tensorflow-Eager) shows a simple example for training a semantic segmentation model using tensorflow with eager mode.
















