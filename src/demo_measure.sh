#!/bin/bash

arch="${1:-hourglass}"
exp_id=$2
model="${3:-../exp/arc/${exp_id}/model_last.pth}"
png_dir="${4:-../data/arc/val}"
python ./demo_measure.py arc --arch ${arch} --demo ${png_dir} --load_model ${model}
