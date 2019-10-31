import os
import glob

import shlex
import subprocess
import shutil

from multiprocessing import Pool
import multiprocessing as multi

exp_dir = '../exp/arc/arc_dla_34'
if not os.path.exists(exp_dir):
    print(f"{exp_dir} doesn't exist")
    exit(-1)

model = f'{exp_dir}/model_last.pth'
if not os.path.exists(model):
    print(f"{model} doesn't exist")
    exit(-1)

result_dir = f'{exp_dir}/demo'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

png_dir = '../data/arc/val'
if not os.path.exists(png_dir):
    print(f"{png_dir} doesn't exist")
    exit(-1)

py_exe = 'python ./demo.py'
arch = 'dla_34'

def demo(png_path):
    python_script = f"{py_exe} arc --arch {arch} " \
                    f"--demo {png_path} --load_model {model}"
    subprocess.run(shlex.split(python_script))
    png_file = os.path.basename(png_path)
    root, _ = os.path.splitext(png_file)
    txt_file = root + '.txt'
    shutil.move(png_file, f'{result_dir}/')
    shutil.move(txt_file, f'{result_dir}/')

p = Pool(multi.cpu_count()//2)
p.map(demo, glob.glob(f'{png_dir}/*.png'))
p.close()
