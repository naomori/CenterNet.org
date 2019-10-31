#python test.py arc --dataset arc --exp_id arc_hg --arch hourglass --num_workers 4 --gpus 1 --keep_res --load_model ../exp/arc/arc_hg/model_last.pth 

python test.py arc --dataset arc --exp_id arc_hg --arch hourglass --gpus 1 --keep_res --resume 
python test.py arc --dataset arc --exp_id arc_hg --arch hourglass --gpus 1 --keep_res --resume --flip_test 
python test.py arc --dataset arc --exp_id arc_hg --arch hourglass --gpus 1 --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5
