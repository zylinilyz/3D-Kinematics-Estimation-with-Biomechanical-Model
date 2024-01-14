import os
import sys

# the directoy of the current script
cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{cwd}/../src/')

dataset_mode = 'data_osim_vid'
model = 'BMRV_OpenSim_sequence'
home_dir = f'{cwd}/../src/'
train_data_dirname = f'{cwd}/../data/train_data/train_data_yolo/'
train_code = f'{home_dir}/train_vid.py'

name = 'bmvr_model_spatialtemporal'
checkpoints_dir = f'{cwd}/../outputs/'
pretrained_checkpoint_path = f'{cwd}/../checkpoints/spatial_model_epoch_14.pth'

lr_G = 0.00005
lr_G_temporal = 0.00005

num_points = 500
reduced_dim = 32
num_view = 2
hg_heads = 128
point_weight = 100

free_hg = 0
use_pretrained = 2
seg_size = 32

chunk_length = 64
temporal_only = 1

cmd = f'python {train_code} --dataset_mode {dataset_mode} \
              --model {model} \
              --nepochs_no_decay 50 --nepochs_decay 50 \
              --lr_G {lr_G} \
              --lr_G_temporal {lr_G_temporal} \
              --name {name} \
              --batch_size 2 \
              --display_freq_s 3600 \
              --save_latest_freq_s 21600  \
              --num_view {num_view} \
              --num_points {num_points} \
              --reduced_dim {reduced_dim} \
              --home_dir {home_dir} \
              --checkpoints_dir {checkpoints_dir} \
              --train_data_dirname {train_data_dirname}\
              --chunk_length {chunk_length}\
              --temporal_only {temporal_only}\
              --hg_heads {hg_heads}\
              --point_weight {point_weight}\
              --free_hg {free_hg}\
              --use_pretrained {use_pretrained}\
              --pretrained_checkpoint_path {pretrained_checkpoint_path}\
              --seg_size {seg_size}'

os.system(cmd)
