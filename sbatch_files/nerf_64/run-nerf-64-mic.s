#!/bin/bash
#
#SBATCH --job-name=nerf-64-mic
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
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
python run_nerf.py --config configs/mic.txt --netwidth 64 --netwidth_fine 64
