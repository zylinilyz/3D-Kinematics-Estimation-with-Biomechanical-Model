import os
import sys

# the directoy of the current script
cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{cwd}/../src/')

home_dir = f'{cwd}/../src/'
names = ['spatial', 'spatialtemporal']

test_codes_key = [ 'frame', 'seq']

dataset = 'ODAH'

for name, key in zip(names, test_codes_key):
    
    test_code = f'{cwd}/1_test_datasets_{key}.py'
    
    if key == 'frame':
        model = 'BMRV_OpenSim_image'
        pretrained_checkpoint_path = f'{cwd}/../checkpoints/spatial_model_epoch_14.pth'
    else:
        model = 'BMRV_OpenSim_sequence'
        pretrained_checkpoint_path = f'{cwd}/../checkpoints/spatial_temporal_model_epoch_5.pth'

    exp_name = str.split(name, '_')[0]

    src_dir = f'{cwd}/../data/test_data/'
    out_dir = f'{cwd}/../test_outputs/'

    num_points = 500
    reduced_dim = 32
    hg_heads = 128
    num_view = 2
    chunk_length = 64

    cmd = f'python {test_code}\
                --model {model} \
                --name {name} \
                --batch_size 1 \
                --num_view {num_view} \
                --num_points {num_points} \
                --reduced_dim {reduced_dim} \
                --pretrained_checkpoint_path {pretrained_checkpoint_path} \
                --chunk_length {chunk_length}\
                --hg_heads {hg_heads}\
                --src_dir {src_dir}\
                --out_dir {out_dir}\
                --exp_name {exp_name}\
                --dataset {dataset}\
                --home_dir {home_dir}'

    os.system(cmd)