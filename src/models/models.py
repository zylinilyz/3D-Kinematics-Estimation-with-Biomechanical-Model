import os
import torch
from torch.optim import lr_scheduler

class ModelsFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(model_name, *args, **kwargs):
        model = None
        
        # INPUT = IMAGES:
        if model_name == 'BMRV_OpenSim_sequence':
            from .BMRV_OpenSim_sequence import Model
            model = Model(*args, **kwargs)
            
        elif model_name == 'BMRV_OpenSim_image':
            from .BMRV_OpenSim_image import Model
            model = Model(*args, **kwargs)
        else:
            raise ValueError("Model %s not recognized." % model_name)

        print("Model %s was created" % model_name)
        # summary model
        print('--------------------------------------')
        print('Model_all parameters : {}'.format(sum(p.numel() for p in model._img_encoder.parameters() if p.requires_grad)))
        print('--------------------------------------')
        print('---model_feat parameters : {}'.format(sum(p.numel() for p in model._img_encoder.image_filter.parameters() if p.requires_grad)))
        print('--------------------------------------')
        if model_name in ['BMRV_OpenSim_sequence_3d']:
            print('---model_frame2seq parameters : {}'.format(sum(p.numel() for p in model._img_encoder.frame2seq.parameters() if p.requires_grad)))
        print('--------------------------------------')
        print('---model_fc1 parameters : {}'.format(sum(p.numel() for p in model._img_encoder.point2osim.fc1.parameters() if p.requires_grad)))
        print('---model_fc2 parameters : {}'.format(sum(p.numel() for p in model._img_encoder.point2osim.fc2.parameters() if p.requires_grad)))
        print('---model_fc3 parameters : {}'.format(sum(p.numel() for p in model._img_encoder.point2osim.fc3.parameters() if p.requires_grad)))
        print('---model_fc4 parameters : {}'.format(sum(p.numel() for p in model._img_encoder.point2osim.fc4.parameters() if p.requires_grad)))
        print('---model_fc5 parameters : {}'.format(sum(p.numel() for p in model._img_encoder.point2osim.fc5.parameters() if p.requires_grad)))
        print('---model_fc6 parameters : {}'.format(sum(p.numel() for p in model._img_encoder.point2osim.fc6.parameters() if p.requires_grad)))
        
        if model_name in ['BMRV_OpenSim_image_new']:
            print('---model_fc7 parameters : {}'.format(sum(p.numel() for p in model._img_encoder.point2osim.fc7.parameters() if p.requires_grad)))
            print('---model_fc8 parameters : {}'.format(sum(p.numel() for p in model._img_encoder.point2osim.fc8.parameters() if p.requires_grad)))
            print('---model_fc9 parameters : {}'.format(sum(p.numel() for p in model._img_encoder.point2osim.fc9.parameters() if p.requires_grad)))
            
        print('--------------------------------------')
        if model._img_encoder.heads is not None:
            print('---model_hgheads parameters : {}'.format(sum(p.numel() for p in model._img_encoder.heads.parameters() if p.requires_grad))) 
            print('--------------------------------------')
        
        return model


class BaseModel(object):

    def __init__(self, opt):
        self._name = 'BaseModel'

        self._opt = opt
        self._is_train = opt.is_train

        self._save_dir = os.path.join(self._opt.checkpoints_dir, self._opt.name
                + '_InputView%s'%(self._opt.num_view))
        #print(self._save_dir)

    @property
    def name(self):
        return self._name

    @property
    def is_train(self):
        return self._is_train

    def set_input(self, input):
        assert False, "set_input not implemented"

    def set_train(self):
        assert False, "set_train not implemented"

    def set_eval(self):
        assert False, "set_eval not implemented"

    def forward(self, keep_data_for_visuals=False):
        assert False, "forward not implemented"

    # used in test time, no backprop
    def test(self):
        assert False, "test not implemented"

    def get_image_paths(self):
        return {}

    def optimize_parameters(self):
        assert False, "optimize_parameters not implemented"

    def get_current_visuals(self):
        return {}

    def get_current_errors(self):
        return {}

    def get_current_scalars(self):
        return {}

    def save(self, label):
        assert False, "save not implemented"

    def load(self):
        assert False, "load not implemented"

    def _save_optimizer(self, optimizer, optimizer_label, epoch_label):
        save_filename = 'opt_epoch_%s_id_%s.pth' % (epoch_label, optimizer_label)
        save_path = os.path.join(self._save_dir, save_filename)
        torch.save(optimizer.state_dict(), save_path)
        
    def _save_lr_scheduler(self, lr_scheduler, scheduler_label, epoch_label):
        save_filename = 'lr_epoch_%s_id_%s.pth' % (epoch_label, scheduler_label)
        save_path = os.path.join(self._save_dir, save_filename)
        torch.save(lr_scheduler.state_dict(), save_path)

    def _load_optimizer(self, optimizer, optimizer_label, epoch_label):
        load_filename = 'opt_epoch_%s_id_%s.pth' % (epoch_label, optimizer_label)
        load_path = os.path.join(self._save_dir, load_filename)
        assert os.path.exists(
            load_path), 'Weights file not found. Have you trained a model!? We are not providing one {}'.format(load_path)

        optimizer.load_state_dict(torch.load(load_path, map_location='cpu'))
        print('loaded optimizer: %s' % load_path)
        
    def _load_lr_scheduler(self, lr_scheduler, scheduler_label, epoch_label):
        load_filename = 'opt_epoch_%s_id_%s.pth' % (epoch_label, scheduler_label)
        load_path = os.path.join(self._save_dir, load_filename)
        assert os.path.exists(
            load_path), 'Weights file not found. Have you trained a model!? We are not providing one {}'.format(load_path)

        lr_scheduler.load_state_dict(torch.load(load_path, map_location='cpu'))
        print('loaded lr_scheduler: %s' % load_path)

    def _save_network(self, network, network_label, epoch_label):
        save_filename = 'net_epoch_%s_id_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self._save_dir, save_filename)
        torch.save(network.state_dict(), save_path)
        print('saved net: %s' % save_path)

    def _load_network(self, network, network_label, epoch_label):
        load_filename = 'net_epoch_%s_id_%s.pth' % (epoch_label, network_label)
        load_path = os.path.join(self._save_dir, load_filename)
        assert os.path.exists(
            load_path), 'Weights file not found. Have you trained a model!? We are not providing one  {}'.format(load_path)

        network.load_state_dict(torch.load(load_path, map_location='cpu'))
        print('loaded net: %s' % load_path)

    def update_learning_rate(self):
        pass

    def print_network(self, network):
        num_params = 0
        for param in network.parameters():
            num_params += param.numel()
        print(network)
        print('Total number of parameters: %d' % num_params)

    def _get_scheduler(self, optimizer, opt):
        if opt.lr_policy == 'lambda':
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
                return lr_l
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif opt.lr_policy == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
        elif opt.lr_policy == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
        return scheduler
