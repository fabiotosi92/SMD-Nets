import numpy as np
import random

def random_crop(left, right, gt_left, gt_right, crop_width, crop_height):
    # randomly crop images
    x = random.randint(0, left.shape[1] - crop_width)
    y = random.randint(0, left.shape[0] - crop_height)

    left = left[y:y + crop_height, x:x + crop_width, :]
    right = right[y:y + crop_height, x:x + crop_width, :]
    gt_left = gt_left[y:y + crop_height, x:x + crop_width, :]
    gt_right = gt_right[y:y + crop_height, x:x + crop_width, :]

    return left, right, gt_left, gt_right

def flip_lr(left, right, gt_left, gt_right, prob=0.5):
    # randomly flip lr both the left and right images + switch between them
    cond = np.random.uniform(0, 1, 1) < prob
    left_lr = np.fliplr(right) if cond > prob else left
    right_lr = np.fliplr(left) if cond > prob else right
    gt_left_lr = np.fliplr(gt_right) if cond > prob else gt_left
    gt_right_lr = np.fliplr(gt_left) if cond > prob else gt_right
    return left_lr, right_lr, gt_left_lr, gt_right_lr

def flip_ud(left, right, gt_left, gt_right, prob=0.5):
    # randomly flip ud both the left and right images
    cond = np.random.uniform(0, 1, 1) < prob
    left_ud = np.flipud(left) if cond > prob else left
    right_ud = np.flipud(right) if cond > prob else right
    gt_left_ud = np.flipud(gt_left) if cond > prob else gt_left
    gt_right_ud = np.flipud(gt_right) if cond > prob else gt_right
    return left_ud, right_ud, gt_left_ud, gt_right_ud

def color_aug(left, right,
              gamma_low=0.8, gamma_high=1.2,
              brightness_low=0.5, brightness_high=1.2,
              color_low=0.8, color_high=1.2, prob=0.5):
    if np.random.uniform(0, 1, 1) < prob:
        # randomly shift gamma
        random_gamma = np.random.uniform(gamma_low, gamma_high)
        left_aug = left ** random_gamma
        right_aug = right ** random_gamma

        # randomly shift brightness
        random_brightness = np.random.uniform(brightness_low, brightness_high)
        left_aug = left_aug * random_brightness
        right_aug = right_aug * random_brightness

        # randomly shift color
        random_colors = np.random.uniform(color_low, color_high, 3)
        left_aug *= random_colors
        right_aug *= random_colors

        # saturate
        left = np.clip(left_aug, 0, 255)
        right = np.clip(right_aug, 0, 255)

    return left, right


