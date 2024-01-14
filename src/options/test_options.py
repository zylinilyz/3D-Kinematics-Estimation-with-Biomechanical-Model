from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self._parser.add_argument('--out_dir', type=str, default='./output', help='output path')
        
        self._parser.add_argument('--exp_name', type=str, default='', help='experiment name')
        self._parser.add_argument('--dataset', type=str, default='HumanEva', help='dataset name')
        self._parser.add_argument('--src_dir', type=str, default='', help='dataset dir')
        

        self.is_train = False
