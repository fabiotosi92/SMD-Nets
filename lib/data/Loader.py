from torch.utils.data import Dataset
import os
import torch
import random
from ..utils import *
from .augmentation import *

class Loader(Dataset):
    def __init__(self, opt, phase='train'):
        self.opt = opt
        self.root = self.opt.dataroot
        self.phase = phase
        self.num_sample_inout = self.opt.num_sample_inout
        self.listfile = self.opt.training_file if self.phase == "train" \
                        else self.opt.testing_file
        self.scenes = self.get_scenes()
        
        if self.phase == "train":
            self.__init_grid()

        if self.opt.mode == 'active':
            self.pattern = img_loader(self.opt.pattern_path, mode=self.opt.mode)

    def __len__(self):
        return len(self.scenes)

    def __init_grid(self):
        nu = np.linspace(0, self.opt.crop_width - 1, self.opt.crop_width)
        nv = np.linspace(0, self.opt.crop_height - 1, self.opt.crop_height)
        u, v = np.meshgrid(nu, nv)

        self.u = u.flatten()
        self.v = v.flatten()

    def __get_coords(self, gt):
        #  Subpixel coordinates
        u = self.u + np.random.random_sample(self.u.size)
        v = self.v + np.random.random_sample(self.v.size)

        # Nearest neighbor
        d = gt[np.clip(np.rint(v).astype(np.uint16), 0, self.opt.crop_height-1),
                 np.clip(np.rint(u).astype(np.uint16), 0, self.opt.crop_width-1)]

        # Remove invalid disparitiy values
        u = u[np.nonzero(d)]
        v = v[np.nonzero(d)]
        d = d[np.nonzero(d)]

        return np.stack((u, v, d), axis=-1)

    def sampling(self, render_data):
        gt = render_data['gt_disp'].data.numpy().squeeze()

        if self.opt.sampling == "random":
            random_points = self.__get_coords(gt)
            idx = np.random.choice(random_points.shape[0], self.num_sample_inout)
            points = random_points[idx, :]

        elif self.opt.sampling == "dda":
            edges = get_boundaries(gt, dilation=self.opt.dilation_factor)
            random_points = self.__get_coords(gt * (1. - edges))
            edge_points = self.__get_coords(gt * edges)

            # if edge points exist
            if edge_points.shape[0]>0:

                # Check tot num of edge points
                cond = edges.sum()//2 -  self.num_sample_inout//2 < 0
                tot= (self.num_sample_inout - int(edges.sum())//2, int(edges.sum())//2) if cond else \
                     (self.num_sample_inout//2, self.num_sample_inout//2)

                idx = np.random.choice(random_points.shape[0], tot[0])
                idx_edges = np.random.choice(edge_points.shape[0], tot[1])
                points = np.concatenate([edge_points[idx_edges, :], random_points[idx, :]], 0)
            # use uniform sample otherwise
            else:
                random_points = self.__get_coords(gt)
                idx = np.random.choice(random_points.shape[0], self.num_sample_inout)
                points = random_points[idx, :]

        return {'samples': np.array(points.T, dtype=np.float32),
                'labels': np.array(points[:,2:3].T, dtype=np.float32)}

    def get_scenes(self):
        with open(self.listfile) as f:
            scenes = f.readlines()
            scenes = [line.rstrip().split() for line in scenes]
        return scenes

    def get_passive_imgs(self, scene):
        '''
        scene[0]: relative path to the left RGB image
        scene[1]: relative path to the right RGB image
        scene[2]: relative path to the ground truth disparity aligned with the left image
        scene[3]: relative path to the ground truth disparity aligned with the right image
        '''
        left = img_loader(os.path.join(self.root, scene[0]), mode=self.opt.mode)
        right = img_loader(os.path.join(self.root, scene[1]), mode=self.opt.mode)

        # scaled GT based on the desired res
        gt_left = gt_loader(os.path.join(self.root, scene[2])) * self.opt.aspect_ratio
        gt_right = gt_loader(os.path.join(self.root, scene[3])) * self.opt.aspect_ratio

        if self.phase == "train":
            left, right, gt_left, gt_right = random_crop(left, right, gt_left, gt_right,
                                                         self.opt.crop_width, self.opt.crop_height)

            # resized rgb images (no GT)
            height, width = (int(left.shape[1] * self.opt.aspect_ratio),
                             int(left.shape[0] * self.opt.aspect_ratio))

            left = cv2.resize(left, (height, width))
            right = cv2.resize(right, (height, width))

            left, right = color_aug(left, right, gamma_low=self.opt.gamma_low, gamma_high=self.opt.gamma_high,
                                                 brightness_low=self.opt.brightness_low, brightness_high=self.opt.brightness_high,
                                                 color_low=self.opt.color_low, color_high=self.opt.color_high)
            left, right, gt_left, gt_right = flip_lr(left, right, gt_left, gt_right)
            left, right, gt_left, gt_right = flip_ud(left, right, gt_left, gt_right)

            left = left.transpose(2,0,1)
            right = right.transpose(2,0,1)

        else:
            height, width = (int(left.shape[1] * self.opt.aspect_ratio),
                             int(left.shape[0] * self.opt.aspect_ratio))

            left = cv2.resize(left, (height, width)).transpose(2,0,1)
            right = cv2.resize(right, (height, width)).transpose(2,0,1)
            
            left, right = pad_imgs(left, right, left.shape[1], left.shape[2])

            left = np.expand_dims(left, 0)
            right = np.expand_dims(right, 0)

            gt_left = np.expand_dims(cv2.resize(gt_left, (height * self.opt.superes_factor, width * self.opt.superes_factor),
                                                interpolation= cv2.INTER_NEAREST), -1).transpose(2,0,1)

        return {
            'left': torch.from_numpy(left.copy()).float(),
            'right': torch.from_numpy(right.copy()).float(),
            'gt_disp': torch.from_numpy(gt_left.copy()).float(),
            'o_shape': torch.from_numpy(np.asarray((height, width)).copy()),
        }

    def get_active_imgs(self, scene):
        '''
        scene[0]: relative path to the active reference image
        scene[1]: relative path to the ground truth disparity 
        '''
        left = img_loader(os.path.join(self.root, scene[0]), mode=self.opt.mode)

        # scaled GT based on the desired res
        gt = gt_loader(os.path.join(self.root, scene[1])) * self.opt.aspect_ratio

        if self.phase == "train":
            x = random.randint(0, left.shape[1] - self.opt.crop_width)
            y = random.randint(0, left.shape[0] - self.opt.crop_height)

            left = left[y:y + self.opt.crop_height, x:x + self.opt.crop_width, :]
            right = self.pattern[y:y + self.opt.crop_height, x:x + self.opt.crop_width, :]
            gt_left = gt[y:y + self.opt.crop_height, x:x + self.opt.crop_width, :]

            # resized rgb images (no GT depth)
            height, width = (int(left.shape[1] * self.opt.aspect_ratio),
                             int(left.shape[0] * self.opt.aspect_ratio))

            left = np.expand_dims(cv2.resize(left, (height, width)), 0)
            right = np.expand_dims(cv2.resize(right, (height, width)), 0)
        else:
            height, width = (int(left.shape[1] * self.opt.aspect_ratio),
                             int(left.shape[0] * self.opt.aspect_ratio))

            left = np.expand_dims(cv2.resize(left, (height, width)), 0)
            right = np.expand_dims(cv2.resize(self.pattern, (height, width)), 0)
            left, right = pad_imgs(left, right, left.shape[1], left.shape[2])

            left = np.expand_dims(left, 0)
            right = np.expand_dims(right, 0)

            gt_left = np.expand_dims(cv2.resize(gt, (height, width), interpolation= cv2.INTER_NEAREST), -1).transpose(2,0,1)

        return {
            'left': torch.from_numpy(left.copy()).float(),
            'right': torch.from_numpy(right.copy()).float(),
            'gt_disp': torch.from_numpy(gt_left.copy()).float(),
            'o_shape': torch.from_numpy(np.asarray((height, width)).copy()),
        }

    def __getitem__(self, index):
        sid = index % len(self.scenes)
        scene = self.scenes[sid]
        res = {'name': os.path.splitext(os.path.basename(scene[0]))[0],
               'name_scene': os.path.dirname(scene[0]),
               'sid': sid}
        render_data = self.get_passive_imgs(scene) if self.opt.mode == "passive" else \
                      self.get_active_imgs(scene)
        res.update(render_data)
        if self.phase == "train":
            sample_data = self.sampling(render_data)
            res.update(sample_data)
        return res
