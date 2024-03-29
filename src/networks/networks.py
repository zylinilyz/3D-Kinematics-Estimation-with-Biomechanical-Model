import torch.nn as nn
import functools

class NetworksFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(network_name, *args, **kwargs):

        if network_name == 'LVD_images':
            from .LVD_images import Network
            network = Network(*args, **kwargs)
        elif network_name == 'BMRV_sequence_network':
            from .BMRV_sequence_network import Network
            network = Network(*args, **kwargs)
        elif network_name == 'BMRV_image_network':
            from .BMRV_image_network import Network
            network = Network(*args, **kwargs)
        else:
            raise ValueError("Network %s not recognized." % network_name)

        print("Network %s was created" % network_name)

        return network


class NetworkBase(nn.Module):
    def __init__(self):
        super(NetworkBase, self).__init__()
        self._name = 'BaseNetwork'

    @property
    def name(self):
        return self._name

    def init_weights(self):
        if  self.point2osim is not None:
            self.point2osim.apply(self._weights_init_fn)
        
        if self.heads is not None:
            self.heads.apply(self._weights_init_fn)
       
        self.apply(self._weights_init_fn)

    def _weights_init_fn(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            try:
                m.weight.data.normal_(0.0, 0.02)
            except:
                for i in m.children():
                    i.apply(self._weights_init_fn)
                return
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0)
        elif classname.find('BatchNorm2d') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.02)
            m.bias.data.fill_(0)

    def _get_norm_layer(self, norm_type='batch'):
        if norm_type == 'batch':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        elif norm_type == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        elif norm_type =='batchnorm2d':
            norm_layer = nn.BatchNorm2d
        else:
            raise NotImplementedError('normalization layer [%s] is not found' % norm_type)

        return norm_layer
