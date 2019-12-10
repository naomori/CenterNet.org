import os
import sys
import glob

import shlex
import subprocess
import shutil

from multiprocessing import Pool
import multiprocessing as multi

py_exe = 'python ./demo_arc.py'

args = sys.argv
arch=args[1]
val_id=args[2]

exp_id=f"{val_id[4:]}"
exp_dir=f"../exp/arc/{exp_id}"
result_name=f"{exp_id[:-3]}"
max_epoch = val_id[-3:]
result_dir_prefix=f"../results/{result_name}"

epochs = [ str(epoch) for epoch in range(200, int(max_epoch)+1) \
                      if (epoch % 100 == 0) or (epoch == int(max_epoch))]

def model_str(x, max):
    if x == max:
        return 'last'
    else:
        return x

models = [model_str(x,max_epoch) for x in epochs]

def demo_arc_script(script_args):
    model,result_dir = script_args
    python_script = f"{py_exe} {arch} {model} {result_dir}"
    subprocess.run(shlex.split(python_script))

p = Pool(1)
p.map(demo_arc_script,
      [(f"{exp_dir}/model_{model}.pth", f"{result_dir_prefix}{epoch}/") for epoch,model in zip(epochs,models)])
p.close()
