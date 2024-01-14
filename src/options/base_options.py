import argparse
import os
from utils import util
import torch

class BaseOptions():
    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self._initialized = False

    def initialize(self):
        self._parser.add_argument('--load_epoch', type=int, default=-1, help='which epoch to load? set to -1 to use latest cached model')
        self._parser.add_argument('--batch_size', type=int, default=12, help='input batch size')
        self._parser.add_argument('--name', type=str, default='experiment_1', help='name of the experiment. It decides where to store samples and models')
        self._parser.add_argument('--dataset_mode', type=str, default='dataset', help='chooses dataset to be used')
        self._parser.add_argument('--model', type=str, default='model', help='model to run[l2_rgb, gan_rgb]')
        self._parser.add_argument('--n_threads_test', default=0, type=int, help='# threads for loading data')
        self._parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self._parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')


        # custom flags
        self._parser.add_argument('--home_dir', type=str, default='./', help='home dir')
        self._parser.add_argument('--train_data_dirname', type=str, default='train_data', help='data path')
        self._parser.add_argument('--num_view', type=int, default=1, help='number of views of each frame')
        self._parser.add_argument('--num_points', type=int, default=6890, help='# of sample points')
        self._parser.add_argument('--reduced_dim', type=int, default=39, help='dim of the reduced point feature')
        
        #self._parser.add_argument('--point_loss', type=int, default=0, help='0: coord and scale, 1: add point loss')
        #self._parser.add_argument('--range_loss', type=int, default=0, help='0: w/o 1: w/')
        
        #self._parser.add_argument('--seq_model_name', type=str, default='lstm', help='lstm or transformer')
        self._parser.add_argument('--chunk_length', type=int, default=3, help='number of frames in a chunk')
        self._parser.add_argument('--temporal_only', type=int, default=0, help='1: temporal only, 0: both spatial and temporal')
        
        #self._parser.add_argument('--vel_loss', type=int, default=0, help='1: add vel as liss, 0: w/o')
        #self._parser.add_argument('--acc_loss', type=int, default=0, help='1: add acc as liss, 0: w/o') 
        #self._parser.add_argument('--reg_acc', type=int, default=0, help='1: add acc for reg, 0: directly use prediction')
        
        self._parser.add_argument('--hg_heads', type=int, default=0, help='1: # of heads, 0: w/o')
        self._parser.add_argument('--point_weight', type=float, default=1, help='weight for position loss')
        #self._parser.add_argument('--angle_degree', type=int, default=0, help='1: loss in degree')
        
        #self._parser.add_argument('--cs_lr', type=int, default=0, help='1: use cosine lr scheduler')
        self._parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer name')
        
        #self._parser.add_argument('--center_pelvis', type=int, default=0, help='1: predict relative to pelvis')
        #self._parser.add_argument('--cos_rot', type=int, default=0, help='1: calculate loss on cos sin domain for pelvis and arms')
        #self._parser.add_argument('--rot_aug', type=int, default=0, help='1: enable rotation along y-zxis during training')
        
        #self._parser.add_argument('--seq_rearrange', type=int, default=0, help='1: rearange feature position')
        #self._parser.add_argument('--temporal_emb', type=int, default=0, help='1: embedding with time')
        #self._parser.add_argument('--scale_reg', type=int, default=0, help='1: add scale regularization term')
        #self._parser.add_argument('--mask_feat', type=int, default=0, help='1: randomly mask input features to the seq network')
        
        
        #self._parser.add_argument('--lcl_loss', type=int, default=0, help='1: use lcl loss')
        #self._parser.add_argument('--sixd_angle', type=int, default=0, help='1: use 6D rotation for pelvis and arm rotation')
        #self._parser.add_argument('--rot_angle_loss', type=int, default=0, help='1: use 6D rotation for pelvis and arm rotation')
        #self._parser.add_argument('--to_cam', type=int, default=0, help='1: prediction toward camera')
        
        self._parser.add_argument('--pretrained_checkpoint_path', type=str, default='', help='pre-trained frame encoder path')
        self._parser.add_argument('--hg_out', type=int, default=256, help='output channel of HG')
        self._parser.add_argument('--free_hg', type=int, default=0, help='if release free_hg params')
        self._parser.add_argument('--use_pretrained', type=int, default=0, help='if use pre-trained model, 0: scratch, 1: lvd, 2: frame-based')
        self._parser.add_argument('--seg_size', type=int, default=32, help='segment size')
        
        #self._parser.add_argument('--data_mode', type=int, default=0, help='0: all, 50: 50, 100: 100')
        
        #self._parser.add_argument('--point_loss_proj', type=int, default=0, help='0: 3D point loss, 1: add projected 2D point loss')
        
        
        self._initialized = True

    def parse(self):
        if not self._initialized:
            self.initialize()
        self._opt = self._parser.parse_args()

        # set is train or set
        self._opt.is_train = self.is_train

        # set and check load_epoch
        self._set_and_check_load_epoch()

        # get and set gpus
        args = vars(self._opt)

        # print in terminal args
        self._print(args)

        # save args to file
        self._save(args)

        return self._opt

    def _set_and_check_load_epoch(self):
        models_dir = os.path.join(self._opt.checkpoints_dir, self._opt.name
                + '_InputView%s'%(self._opt.num_view))
        if os.path.exists(models_dir):
            if self._opt.load_epoch == -1:
                load_epoch = 0
                for file in os.listdir(models_dir):
                    if file.startswith("net_epoch_"):
                        load_epoch = max(load_epoch, int(file.split('_')[2]))
                self._opt.load_epoch = load_epoch
            else:
                found = False
                for file in os.listdir(models_dir):
                    if file.startswith("net_epoch_"):
                        found = int(file.split('_')[2]) == self._opt.load_epoch
                        if found: break
                assert found, 'Model for epoch %i not found' % self._opt.load_epoch
        else:
            assert self._opt.load_epoch < 1, 'Model for epoch %i not found' % self._opt.load_epoch
            self._opt.load_epoch = 0

    def _print(self, args):
        print('------------ Options -------------')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Device', device)
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

    def _save(self, args):
        expr_dir = os.path.join(self._opt.checkpoints_dir, self._opt.name
                + '_InputView%s'%(self._opt.num_view))
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt_%s.txt' % ('train' if self.is_train else 'test'))
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print('Device', device)
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
