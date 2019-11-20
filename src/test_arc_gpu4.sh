# $1: arch: hourglass, dla_34, resdcn_101, resdcn_18
# $2: exp_id

# arch: hourglass, dla_34, resdcn_101, resdcn_18
arch=$1
exp_id=$2

export CUDA_VISIBLE_DEVICES=0,1,2,3
# test
python test.py arc --dataset arc --arch ${arch} --exp_id ${exp_id} --keep_res \
        --resume
# flip test
python test.py arc --dataset arc --arch ${arch} --exp_id ${exp_id} --keep_res \
        --resume --flip_test
# multi scale test
python test.py arc --dataset arc --arch ${arch} --exp_id ${exp_id} --keep_res \
        --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5
