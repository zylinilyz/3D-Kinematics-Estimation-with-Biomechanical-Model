# 3D-Kinematics-Estimation-with-Biomechanical-Model
Implementation of [3D Kinematics Estimation from Video with a Biomechanical Model and Synthetic Training Data](https://arxiv.org/abs/2402.13172), CVPR workshop 2024.

## Requirements
```
conda create python=3.9 -n py39
conda activate py39
conda install pytorch=1.13.1=py3.9_cuda11.6_cudnn8.3.2_0 torchvision=0.14.1=py39_cu116  pytorch-cuda=11.6 opencv -c pytorch -c nvidia
conda install -c conda-forge scikit-learn scikit-image tqdm matplotlib pandas tensorboardX
pip install mediapipe trimesh imgaug timm einops bs4 lxml ultralytics==8.0.99
pip install ultralytics==8.0.99
```

## Pre-trained models
Please put the pre-trained model under the directory 'checkpoints'.
* Spatial model: [download here](https://drive.google.com/file/d/1uYR4fssAdrtCG0sS03wYTi7XOlFCWyQQ/view?usp=sharing)
* Spatio-temporal model: [download here](https://drive.google.com/file/d/1M2bDvhiYHM4KBAKFGjY5D7nU2-quydYu/view?usp=share_link)

## Training
### Pre-processing
Please specify the source directory of the training data and the output root directory for the pre-processed training data. 
```
python ./training_scripts/0_preprocess_train_data.py --data_dir  ./data/train_data/ --out_dir ./data/train_data/
```
Remember to prepare your validation data. The final folder structure for training should follow:
```bash
data
└── train_data
    ├── train_data_yolo
    │   ├── coords
    │   ├── images
    │   ├── points
    │   └── scales
    └── val_data_yolo
        ├── coords
        ├── images
        ├── points
        └── scales
```

### Train spatial model
We first train the frame feature encoder without considering temporal information. Please download the pre-trained LVD_images_SMPL model [here](https://github.com/enriccorona/LVD/blob/main/README.md?plain=1), and put it under the directory 'checkpoints'.
```
python ./training_scripts/1_run_train_spatial.py
```
### Train spatio-temporal model
Next, we train the spatiotemporal model for spatio-temporal feature refinement. Please make sure the checkpoint of the spatial model is correctly specified in '2_run_train_spatialtemporal.py'.
```
python ./training_scripts/2_run_train_spatialtemporal.py
```
## Testing
### Pre-processing
Please specify the directory for the testing data and the output root directory for the pre-processed testing data.
```
python ./testing_scripts/0_prepare_test_data.py --data_dir ./data/test_data/ --out_dir ./data/test_data/
```
### Run testing
```
python ./testing_scripts/run_test.py
```
### Post-processing
The last step is to integrate all the predictions to reconstruct the OpenSim model. The outputs include the body segment scales for the subject, the joint angles, and the joint positions estimated from the input video. 
```
python ./testing_scripts/2_combine_mot_scale_point.py
```

## Acknowledgements
This implementation is based on [LVD](https://github.com/enriccorona/LVD/blob/main/README.md?plain=1).

## Citation
```
@inproceedings{lin20243d,
  title={3D kinematics estimation from video with a biomechanical model and synthetic training data},
  author={Lin, Zhi-Yi and Lyu, Bofan and Fernandez, Judith Cueto and Van Der Kruk, Eline and Seth, Ajay and Zhang, Xucong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1441--1450},
  year={2024}
}
```


