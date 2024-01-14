import torch.nn as nn
import torch


class HGHeads(nn.Module):
    def __init__(self, in_dim = 256, out_dim = 64):
        super(HGHeads, self).__init__()
        self.heads = []
            
        self.head1 = torch.nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)
        self.head2 = torch.nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)
        self.head3 = torch.nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)
        self.head4 = torch.nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)
    
        
    def forward(self, im_feat_list):
        
        im_feat_list
        y = []
        y.append(self.head1(im_feat_list[0]))
        y.append(self.head2(im_feat_list[1]))
        y.append(self.head3(im_feat_list[2]))
        y.append(self.head4(im_feat_list[3]))
        
        return y
