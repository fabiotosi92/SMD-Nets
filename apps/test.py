import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader
from lib.options import BaseOptions
from lib.data import Loader
from lib.model import *
from lib.evaluation_utils import *

# get options
opt = BaseOptions().parse()

def test(opt):
    # set cuda
    cuda = torch.device('cuda:%d' % opt.gpu_id)

    test_dataset = Loader(opt, phase='test')

    test_data_loader = DataLoader(test_dataset,
                                   batch_size=1, shuffle=False,
                                   num_workers=opt.num_threads, pin_memory=opt.pin_memory)
    print('test data size: ', len(test_data_loader))

    # create net
    net = SMDHead(opt).to(device=cuda)

    def set_eval():
        net.eval()

    # load checkpoints
    if opt.load_checkpoint_path is not None:
        print('loading weights ...', opt.load_checkpoint_path)
        net.load_state_dict(torch.load(opt.load_checkpoint_path, map_location=cuda)['state_dict'])

    os.makedirs('%s' % (opt.results_path), exist_ok=True)

    with torch.no_grad():
        set_eval()
        validation(opt, net, cuda, test_dataset, num_gen_test=len(test_dataset), save_imgs=True)

if __name__ == '__main__':
    test(opt)