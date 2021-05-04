#!bin/bash

#args
mode="passive"
name="train"

# setup
setup=stereo
dataset=UnrealStereo4K

# path
results_path=./output/$dataset/$setup/$mode/
checkpoints_path=./checkpoints/$dataset/$setup/$mode/$name

# network
backbone="PSMNet"

# filenames
training_file=./filenames/train.txt
testing_file=./filenames/val.txt

# training settings
batch_size=4
num_epoch=200
learning_rate=1e-4
gamma=0.1

# sampling strategy (random, dda)
sampling=dda
num_sample_inout=50000

# output_representation
output_representation="bimodal"

# extras
extras="--pin_memory"
#extras="--continue_train"

# datasets
if [ $dataset == "UnrealStereo4K" ];then

    dataroot=/media/Storage/Datasets/UnrealStereo4K
    aspect_ratio=0.25
    crop_width=2048
    crop_height=1536

elif [ $dataset == "KITTI" ];then

    dataroot=/media/Storage/Datasets/KITTI/2015
    aspect_ratio=1.
    crop_width=896
    crop_height=256
fi

python apps/train.py --dataroot $dataroot \
                     --checkpoints_path $checkpoints_path \
                     --training_file $training_file \
                     --testing_file $testing_file \
                     --results_path $results_path \
                     --mode $mode \
                     --name $name \
                     --batch_size $batch_size \
                     --num_epoch $num_epoch \
                     --learning_rate $learning_rate \
                     --gamma $gamma \
                     --crop_height $crop_height \
                     --crop_width $crop_width \
                     --num_sample_inout $num_sample_inout \
                     --aspect_ratio $aspect_ratio \
                     --sampling $sampling \
                     --output_representation $output_representation \
                     --backbone $backbone \
                     $extras


