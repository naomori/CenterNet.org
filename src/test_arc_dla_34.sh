#python test.py arc --dataset arc --exp_id arc_dla_34 --arch dla_34 --keep_res  --num_workers 4 --load_model ../exp/arc/arc_dla_34/model_last.pth
python test.py arc --dataset arc --exp_id arc_dla_34 --keep_res --resume
python test.py arc --dataset arc --exp_id arc_dla_34 --keep_res --resume --flip_test
python test.py arc --dataset arc --exp_id arc_dla_34 --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5