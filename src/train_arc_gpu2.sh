# $1: arch: hourglass, dla_34, resdcn_101, resdcn_18
# $2: exp_id
# $3: batch size
# #4: num_epochs

# arch: hourglass, dla_34, resdcn_101, resdcn_18
arch=$1
exp_id=$2

export CUDA_VISIBLE_DEVICES=1,2
case $arch in
  # Hourglass
  "hourglass")
    batch_size="${3:-10}"
    num_epochs="${4:-140}"
    python main.py arc --dataset arc --arch ${arch} \
      --exp_id ${exp_id} --batch_size ${batch_size} --master_batch 4 \
      --lr 2.5e-4 --num_epochs ${num_epochs} \
      --load_model ../models/ExtremeNet_500000.pth \
      --gpus 1,2 ;;
  # Deep Layer Aggregation
  "dla_34")
    batch_size="${3:-10}"
    num_epochs="${4:-230}"
    python main.py arc --dataset arc --arch ${arch} \
      --exp_id ${exp_id} --batch_size ${batch_size} --master_batch 9 \
      --lr 5e-4 --gpus 1,2 --num_workers 4 \
      --num_epochs ${num_epochs} --lr_step 180,210 ;;
  # ResDCN
  "resdcn_101")
    batch_size="${3:-10}"
    num_epochs="${4:-140}"
    python main.py arc --dataset arc --arch ${arch} \
      --exp_id ${exp_id} --batch_size ${batch_size} --master_batch 5 \
      --num_epochs ${num_epochs} --lr 3.75e-4 --gpus 1,2 --num_workers 4 ;;
  "resdcn_18")
    batch_size="${3:-10}"
    num_epochs="${4:-140}"
    python main.py ctdet --dataset arc --arch ${arch} \
      --exp_id ${exp_id} --batch_size ${batch_size} --master_batch 18 \
      --num_epochs ${num_epochs} --lr 5e-4 --gpus 1,2 --num_workers 4 ;;
  *) echo "unknown arch: ${arch}" ;;
esac
