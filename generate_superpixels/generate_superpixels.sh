comando='cli' # superpixel method

data=$1/images
data_train=$data/train
data_test=$data/test
pathname=$data

# name of the dataset
name="$(echo $(basename $(dirname $data)))"



# output paths
out_test=$name/superpixels/test
out_train=$name/superpixels/train

mkdir -p $out_train
mkdir -p $out_test

shift

#For each value (N), creates segmentations of N segments
for i in $@; do
 	./bin/$comando --input $data_train --output $out_train/superpixels_$i --contour --csv  --superpixels   $i
 	./bin/$comando --input $data_test --output $out_test/superpixels_$i --contour --csv  --superpixels   $i
 done
 
 echo GENERATION COMPLETED
