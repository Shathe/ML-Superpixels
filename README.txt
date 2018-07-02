
## Generate a sparse ground-truth from a fully-labled ground-truth

This step generates images with some labeled pixels.

The dataset has to have this folder structure:
-dataset
	-images 
		-train
		-test
	-labels
		-train
		-test

The code that executes this step:
python generate_sparse/generate_sparse.py --dataset ../Datasets/camvid --n_labels 500  --image_format png --default_value 255

Every sparse labeled image will have [n_labels] number of labeled pixels. The rest pixels will have the [default_value] value
The output folder will have the same name as the [dataset]: dataset/sparse_GT/train and dataset/sparse_GT/test




## Generate superpixels


sh generate_superpixels/generate_superpixels.sh ../Datasets/camvid/
The first argument is the dataset path


## Generate augmented ground-truth

python generate_augmented_GT/generate_augmented_GT.py --dataset ./camvid/





# Evaluation


# camvid llega a pixel acc 91.95 mean pixel acc 76.91   miou 65.05 ?
# comentar codigo bien