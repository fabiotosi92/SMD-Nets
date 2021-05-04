import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import json
import torch
from torch.utils.data import DataLoader
from lib.options import BaseOptions
from lib.evaluation_utils  import *
from lib.data import *
from lib.model import *
from tensorboardX import SummaryWriter
import time

# get options
opt = BaseOptions().parse()

def wif(id):
    process_seed = torch.initial_seed()
    base_seed = process_seed - id
    ss = np.random.SeedSequence([id, base_seed])
    np.random.seed(ss.generate_state(4))
    
def train(opt):
    # set cuda
    cuda = torch.device('cuda:%d' % opt.gpu_id)

    train_dataset = Loader(opt, phase='train')
    test_dataset = Loader(opt, phase='test')

    # create data loader
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=opt.batch_size, 
                                   shuffle=not opt.serial_batches,
                                   num_workers=opt.num_threads, 
                                   pin_memory=opt.pin_memory,
                                   worker_init_fn=wif)

    print('train data size: ', len(train_data_loader))

    test_data_loader = DataLoader(test_dataset,
                                   batch_size=1, shuffle=False,
                                   num_workers=opt.num_threads, pin_memory=opt.pin_memory)
    print('test data size: ', len(test_data_loader))

    # create net
    net = SMDHead(opt).to(device=cuda)

    optimizer = torch.optim.Adam(net.parameters(), lr=opt.learning_rate)

    lr = opt.learning_rate

    def set_train():
        net.train()

    def set_eval():
        net.eval()

    # load checkpoints
    if opt.load_checkpoint_path is not None:
        print('loading weights ...', opt.load_checkpoint_path)
        net.load_state_dict(torch.load(opt.load_checkpoint_path, map_location=cuda)['state_dict'])

    if opt.continue_train and (os.path.exists("%s/%s/net_latest"% (opt.checkpoints_path, opt.name)) or
                               os.path.exists("%s/%s/net_epoch_%d"% (opt.checkpoints_path, opt.name, opt.resume_epoch))):
        if opt.resume_epoch < 0:
            model_path = '%s/%s/net_latest' % (opt.checkpoints_path, opt.name)
        else:
            model_path = '%s/%s/net_epoch_%d' % (opt.checkpoints_path, opt.name, opt.resume_epoch)
        print('Resuming from ', model_path)

        net, optimizer, start_epoch, lr = load_ckp(model_path, cuda, net, optimizer)
    else:
        start_epoch = 0

    os.makedirs(opt.checkpoints_path, exist_ok=True)
    os.makedirs(opt.results_path, exist_ok=True)
    os.makedirs('%s/%s' % (opt.checkpoints_path, opt.name), exist_ok=True)
    os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)

    opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
    with open(opt_log, 'w') as outfile:
        outfile.write(json.dumps(vars(opt), indent=2))

    writer = SummaryWriter('%s/%s' % (opt.checkpoints_path, opt.name))
    print('%s/%s' % (opt.checkpoints_path, opt.name))

    tstart = time.time()

    print("* Start Training")
    for epoch in range(start_epoch, opt.num_epoch):
        epoch_start_time = time.time()

        # update learning rate
        lr = adjust_learning_rate(optimizer, epoch, lr, opt.schedule, opt.gamma)

        set_train()
        iter_data_time = time.time()

        train_dataset.is_train = True
        for train_idx, train_data in enumerate(train_data_loader):
            set_train()
            iter_start_time = time.time()
            total_step = epoch * len(train_data_loader) + train_idx

            # retrieve the data
            left = train_data['left'].to(device=cuda)
            right = train_data['right'].to(device=cuda)
            sample = train_data['samples'].to(device=cuda)
            labels = train_data['labels'].to(device=cuda)
            errors = net.forward(left, right, sample, labels=labels)

            total_error = 0
            for error in errors.values():
                total_error+= error

            optimizer.zero_grad()
            total_error.backward()
            optimizer.step()

            iter_net_time = time.time()
            eta = ((iter_net_time - epoch_start_time) / (train_idx + 1)) * len(train_data_loader) - (
                iter_net_time - epoch_start_time)

            if train_idx % opt.freq_plot == 0:
                print(
                    'Name: {0} | Epoch: {1} | {2}/{3} | Err: {4:.06f} | learning rate: {5:.06f} | netT: {7:.05f} | ETA: {8:02d}:{9:02d}'.format(
                        opt.name, epoch, train_idx, len(train_data_loader), total_error, lr, iter_start_time - iter_data_time,
                        iter_net_time - iter_start_time, int(eta // 60), int(eta - 60 * (eta // 60))))

                for k, v in errors.items():
                    writer.add_scalar(k, v, total_step)
                writer.add_scalar('total_loss', total_error, total_step)
                writer.add_scalar('learning_rate', lr, total_step)

            # validation
            if total_step % opt.freq_eval == 0 and not (epoch + train_idx == 0):
                with torch.no_grad():
                    set_eval()
                    res, left_imgs, preds, gts = validation(opt, net, cuda, test_dataset, num_gen_test=opt.num_gen_test)

                    # write images (summary)
                    left_batch = np.zeros((len(left_imgs), left_imgs[0].shape[0], left_imgs[0].shape[1], left_imgs[0].shape[2]))
                    disp_batch = left_batch.copy()
                    gt_batch = left_batch.copy()
                    epe_batch = left_batch.copy()

                    for i, (img, pred, gt) in enumerate(zip(left_imgs, preds,gts)):
                        mask = np.expand_dims((gt > 0).astype(np.uint8), -1)
                        left_batch[i] = img[::-1,:,:]/255.
                        disp_batch[i] = (color_depth_map(pred)).transpose(2,0,1)[::-1,:,:]/255.
                        gt_batch[i] = (color_depth_map(gt) * mask).transpose(2,0,1)[::-1,:,:]/255.
                        epe_batch[i] = (color_error_image(np.abs(gt - pred)) * mask).transpose(2,0,1)[::-1,:,:]/255.

                    writer.add_images('validation/left', left_batch, total_step)
                    writer.add_images('validation/pred_disp', disp_batch, total_step)
                    writer.add_images('validation/gt_disp', gt_batch, total_step)
                    writer.add_images('validation/error', epe_batch, total_step)

                    # write scalars (summary)
                    for key, value in res.items():
                        writer.add_scalar('validation/%s' % key, value, total_step)

            if total_step % opt.freq_save == 0 and total_step != 0:
                checkpoint = {
                    'epoch': epoch,
                    'learning_rate': lr,
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }

                checkpoint_path = '%s/%s/net_latest' % (opt.checkpoints_path, opt.name)
                save_ckp(checkpoint, checkpoint_path)

                checkpoint_path = '%s/%s/net_epoch_%d' % (opt.checkpoints_path, opt.name, epoch)
                save_ckp(checkpoint, checkpoint_path)

        if time.time() - tstart > opt.save_every and opt.save_every > 0:
            checkpoint = {
                'epoch': epoch+1,
                'learning_rate': lr,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            checkpoint_path = '%s/%s/net_latest' % (opt.checkpoints_path, opt.name)
            save_ckp(checkpoint, checkpoint_path)
            torch.cuda.empty_cache()

            exit(3)

    writer.close()

if __name__ == '__main__':
    train(opt)
