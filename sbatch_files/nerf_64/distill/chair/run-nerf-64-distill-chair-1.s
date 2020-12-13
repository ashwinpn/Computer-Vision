#!/bin/bash
#
#SBATCH --job-name=nd-c-1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=96:00:00
#SBATCH --mem=16GB

cd ~/Computer-Vision
module purge
module load cuda/10.1.105
module load gcc/6.3.0
source ~/.bashrc
conda activate NeRF
pip install opencv-python
pip install -r requirements.txt
pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
cd torchsearchsorted
pip install .
cd ..
ls -l
python distill_nerf.py saved_models/chair_64_200000.tar --t_width 64 --s_width 64 --max_epochs 1000000 --s_depth 7 --s_skips 3 --layer_queue "0,1|1,2|2,3|3,4|4,5|5,6|6,7|7,8|8,9|O,O" --plot_path "./plots/chair/1/layer_{}_.png" --save_path "./logs/blender_paper_chair/1/student_model_{}.tar"