#!/bin/bash

# $1: arch: hourglass, dla_34, resdcn_101, resdcn_18
# $2: exp_id
# $3: model
# #4: png directory

arch="${1:-hourglass}"
exp_id=$2
model="${3:-../exp/arc/${exp_id}/model_last.pth}"
png_dir="${4:-../data/arc/val}"
python ./demo_measure.py arc --arch ${arch} --demo ${png_dir} --load_model ${model}
