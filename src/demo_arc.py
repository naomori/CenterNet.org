import os
import sys
import glob

import shlex
import subprocess
import shutil

from multiprocessing import Pool
import multiprocessing as multi

args = sys.argv
arch = args[1] # dla_34, hourglass, resdcn_101, resdcn_18
model = args[2]
if not os.path.exists(model):
    print(f"{model} doesn't exist")
    exit(-1)

result_dir = args[3]
if not os.path.exists(result_dir):
    os.makedirs(result_dir, exist_ok=True)

png_dir = '../data/arc/val'
if not os.path.exists(png_dir):
    print(f"{png_dir} doesn't exist")
    exit(-1)

py_exe = 'python ./demo.py'

def demo(png_path):
    python_script = f"{py_exe} arc --arch {arch} " \
                    f"--demo {png_path} --load_model {model} " \
                    f"--result_dir {result_dir}"
    subprocess.run(shlex.split(python_script))

#p = Pool(multi.cpu_count()//2)
p = Pool(4)
p.map(demo, glob.glob(f'{png_dir}/*.png'))
p.close()
