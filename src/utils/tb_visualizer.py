import numpy as np
import os
import time
from . import util
import cv2
from tensorboardX import SummaryWriter
#from utils.util_opensim_wo_lib import *


class TBVisualizer:
    def __init__(self, opt):
        self._opt = opt
        self._save_path = os.path.join(self._opt.checkpoints_dir, self._opt.name
                + '_InputView%s'%(self._opt.num_view))

        self._log_path = os.path.join(self._save_path, 'loss_log2.txt')
        self._tb_path = os.path.join(self._save_path, 'summary.json')
        self._writer = SummaryWriter(self._save_path)
        
        with open(self._log_path, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def __del__(self):
        self._writer.close()

    def display_current_results(self, visuals, it, is_train, save_visuals=False):
        for label, image_numpy in visuals.items():
            sum_name = '{}/{}'.format('Train' if is_train else 'Test', label)
            #from IPython import embed
            #embed()
            if save_visuals:
                if label in ['input_image', 'input_mask']:
                    try:
                        for i in range(self._opt.num_view):
                            self._writer.add_image(sum_name + '_view%s'%i, image_numpy[i].transpose((2, 0, 1)), it)
                        
                    except:
                        pass
                        #self._writer.add_image(sum_name, image_numpy[np.newaxis], it)

                    if label in ['smpl_prediction', 'smpl_groundtruth' ]:
                        # Export SMPL object
                        mesh_name = os.path.join(self._save_path,
                                                        'event_obj', sum_name, '%08d.obj' % it)
                        if not os.path.exists(os.path.dirname(mesh_name)):
                            os.makedirs(os.path.dirname(mesh_name))
                        image_numpy.export(mesh_name)
                        
                    elif label in ['osim_prediction', 'osim_groundtruth']:
                        # Export SMPL object
                        obj_name = os.path.join(self._save_path,
                                                            'event_obj', sum_name, '%08d_coords.npy' % it)
                        if not os.path.exists(os.path.dirname(obj_name)):
                            os.makedirs(os.path.dirname(obj_name))
                        with open(obj_name, 'wb') as f:
                            np.save(f, image_numpy[0])
                        
                        obj_name = os.path.join(self._save_path,
                                                            'event_obj', sum_name, '%08d_scales.npy' % it)
                        if not os.path.exists(os.path.dirname(obj_name)):
                            os.makedirs(os.path.dirname(obj_name))
                        with open(obj_name, 'wb') as f:
                            np.save(f, image_numpy[1])
                    

                    elif label in ['input_image', 'input_mask']:
                        try:
                            for i in range(self._opt.num_view):
                                img_name = os.path.join(self._save_path,
                                                            'event_imgs', sum_name, '%08d_view%s.png' % (it, i))
                                if not os.path.exists(os.path.dirname(img_name)):
                                    os.makedirs(os.path.dirname(img_name))
        
                                cv2.imwrite(img_name, np.uint8(image_numpy[i][:,:,::-1]*255))
                        except:
                            pass

        self._writer.export_scalars_to_json(self._tb_path)

    def plot_scalars(self, scalars, it, is_train):
        for label, scalar in scalars.items():
            sum_name = '{}/{}'.format('Train' if is_train else 'Test', label)
            self._writer.add_scalar(sum_name, scalar, it)

    def print_current_train_errors(self, epoch, i, iters_per_epoch, errors, t, visuals_were_stored):
        log_time = time.strftime("[%d/%m/%Y %H:%M:%S]")
        visuals_info = "v" if visuals_were_stored else ""
        message = '%s (T%s, epoch: %d, it: %d/%d, t/smpl: %.3fs) ' % (log_time, visuals_info, epoch, i, iters_per_epoch, t)
        for k, v in errors.items():
            message += '%s:%.5f ' % (k, v)

        print(message)
        with open(self._log_path, "a") as log_file:
            log_file.write('%s\n' % message)

    def print_current_validate_errors(self, epoch, errors, t):
        log_time = time.strftime("[%d/%m/%Y %H:%M:%S]")
        message = '%s (V, epoch: %d, time_to_val: %ds) ' % (log_time, epoch, t)
        for k, v in errors.items():
            message += '%s:%.5f ' % (k, v)

        print(message)
        with open(self._log_path, "a") as log_file:
            log_file.write('%s\n' % message)

    def save_images(self, visuals):
        for label, image_numpy in visuals.items():
            image_name = '%s.png' % label
            save_path = os.path.join(self._save_path, "samples", image_name)
            util.save_image(image_numpy, save_path)
