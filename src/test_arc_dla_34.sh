#python test.py arc --dataset arc --exp_id arc_dla_34 --arch dla_34 --keep_res --num_workers 4 --load_model ../exp/arc/arc_dla_34/model_last.pth
python test.py arc --dataset arc --exp_id arc_dla_34 --keep_res --resume --num_workers 6 --load_model ../exp/arc/arc_dla_34/model_last.pth
python test.py arc --dataset arc --exp_id arc_dla_34 --keep_res --resume --num_workers 6 --flip_test --load_model ../exp/arc/arc_dla_34/model_last.pth
python test.py arc --dataset arc --exp_id arc_dla_34 --keep_res --resume --num_workers 6 --flip_test --test_scales 0.5,0.75,1,1.25,1.5 --load_model ../exp/arc/arc_dla_34/model_last.pth
