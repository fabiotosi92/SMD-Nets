import argparse

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # Datasets related
        g_data = parser.add_argument_group('Data')
        g_data.add_argument('--dataroot', type=str, default='./data', help='path to images (data folder)')
        g_data.add_argument('--training_file', type=str, default='./filenames/train.txt', help='path to the training file')
        g_data.add_argument('--testing_file', type=str, default='./filenames/test.txt', help='path to the testing file')
        g_data.add_argument('--pattern_path', type=str, default='./imgLaser.raw', help='path to the pattern')

        # Stereo camera related
        g_data.add_argument('--maxdisp', type=int, default=256, help='max disparity value')
        g_data.add_argument('--mindisp', type=int, default=0, help='min disparity value')
        g_data.add_argument('--aspect_ratio', type=float, default=0.25, help='aspect ratio between RGB and GT images')
        g_data.add_argument('--superes_factor', type=int, default=1, help='scale factor used to estimate disparities at arbitrary spatial resolution')
        g_data.add_argument('--crop_height', type=int, default=512, help='crop height (computed with respect to the ground truth height)')
        g_data.add_argument('--crop_width', type=int, default=512, help='crop width (computed with respect to the ground truth width)')

        # Experiment related
        g_exp = parser.add_argument_group('Experiment')
        g_exp.add_argument('--name', type=str, default='example', help='name of the experiment. It decides where to store samples and models')
        g_exp.add_argument('--mode', type=str, default='passive', choices=['passive','active'])
        g_exp.add_argument('--output_representation', type=str, default='bimodal', choices=['standard','unimodal', 'bimodal'])

        # Training related
        g_train = parser.add_argument_group('Training')
        g_train.add_argument('--gpu_id', type=int, default=0, help='gpu id for cuda')
        g_train.add_argument('--num_threads', default=8, type=int, help='# sthreads for loading data')
        g_train.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        g_train.add_argument('--pin_memory', action='store_true', help='pin_memory')
        
        g_train.add_argument('--batch_size', type=int, default=4, help='input batch size')
        g_train.add_argument('--learning_rate', type=float, default=1e-4, help='adam learning rate')
        g_train.add_argument('--num_epoch', type=int, default=200, help='num epoch to train')
        g_train.add_argument('--freq_plot', type=int, default=100, help='freqency of the error plot')
        g_train.add_argument('--freq_save', type=int, default=500, help='freqency of the save_checkpoints')
        g_train.add_argument('--freq_eval', type=int, default=2000, help='freqency of the validation (in epochs)')
        g_train.add_argument('--save_every', type=float, default=-1, help='exit the job with status 3 after once the time reaches --save_every')
        g_train.add_argument('--num_write_test', type=int, default=5, help='number of images saved on tensorboard')
        g_train.add_argument('--resume_epoch', type=int, default=-1, help='epoch resuming the training')
        g_train.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')

        # Testing related
        g_test = parser.add_argument_group('Testing')
        g_test.add_argument('--test', action='store_true', help='test mode or not')
        g_test.add_argument('--test_folder_path', type=str, default=None, help='the folder of test image')
        g_test.add_argument('--differential_entropy', action='store_true', help='compute the differential entropy of the continuous mixture distribution (test time)')

        # Sampling related
        g_sample = parser.add_argument_group('Sampling')
        g_sample.add_argument('--num_sample_inout', type=int, default=50000, help='# of sampling points')
        g_sample.add_argument('--sampling', type=str, help='type of sampling, random or depth discontinuity aware', default='random',
                            choices=['random', 'dda'])
        g_sample.add_argument('--dilation_factor', type=int, default=10, help='dilation factor for estimated boundaries')

        # Model related
        g_model = parser.add_argument_group('Model')
        g_model.add_argument('--backbone', type=str, help='stereo backbone', default='PSMNet', choices=['PSMNet', 'HSMNet', 'Stereodepth'])

        # Pre-training specify
        g_model.add_argument('--imagenet_pt', action='store_true')

        # for train
        parser.add_argument('--no_residual', action='store_true', help='no skip connection in mlp')
        parser.add_argument('--no_sine', action='store_true', help='no sine as activation function for the MLP')
        parser.add_argument('--schedule', nargs='+', default=[150, 175], help='Decrease learning rate at these epochs.')
        parser.add_argument('--gamma', type=float, default=0.5, help='the learning rate is multiplied by gamma on schedule.')

        # for eval
        parser.add_argument('--num_gen_test', type=int, default=1, help='how many disparities to generate during testing')

        # path
        parser.add_argument('--checkpoints_path', type=str, default='./checkpoints', help='path to save checkpoints')
        parser.add_argument('--load_checkpoint_path', type=str, default=None, help='path to save checkpoints')
        parser.add_argument('--results_path', type=str, default='./results', help='path to save results')

        # aug
        group_aug = parser.add_argument_group('aug')
        group_aug.add_argument('--gamma_low', type=float, default=0.8, help='augmentation gamma - low ')
        group_aug.add_argument('--gamma_high', type=float, default=1.1, help='augmentation gamma - high')
        group_aug.add_argument('--brightness_low', type=float, default=0.5, help='augmentation saturation brightness - low')
        group_aug.add_argument('--brightness_high', type=float, default=1.5, help='augmentation brightness - high')
        group_aug.add_argument('--color_low', type=float, default=0.8, help='augmentation color - low')
        group_aug.add_argument('--color_high', type=float, default=1.2, help='augmentation color - high')

        # special tasks
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        opt = self.gather_options()
        if len(opt.schedule)==1 and isinstance(opt.schedule[0], str):
            opt.schedule = [int(s) for s in opt.schedule[0].split(',')]
        return opt
