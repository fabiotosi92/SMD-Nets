import torch
import numpy as np
import sys
import re
import cv2

# SMD-Head utilities

def laplacian(x, mu, b):
    return 0.5 * np.exp(-(np.abs(mu-x)/b))/b


def differential_entropy(mu0, mu1, sigma0, sigma1, pi0, pi1, n=2000, a=-1., b=2.):
    eps = 1e-6
    f = lambda x: pi0 * laplacian(x, mu0, sigma0) + pi1 * laplacian(x, mu1, sigma1)
    for k in range(1, n):
        x = a + k*(b-a)/n
        fx = f(x)
        mask = fx<eps
        ent_i = fx * np.log(fx)
        ent_i[mask] = 0
        sum = ent_i if k==1 else sum + ent_i
    return -(b - a)*(f(a)/2 + f(b)/2 + sum)/n


def scale_coords(points, max_length):
    return torch.clamp(2 * points/(max_length-1.)- 1., -1., 1.)


def to_numpy(tensor):
    return tensor.squeeze().detach().cpu().numpy()


def interpolate(feat, uv):
    uv = uv.transpose(1, 2)
    uv = uv.unsqueeze(2)
    samples = torch.nn.functional.grid_sample(feat, uv)
    return samples[:, :, :, 0]


def shift_2d_replace(data, dx, dy, constant=False):
    shifted_data = np.roll(data, dx, axis=1)
    if dx < 0:
        shifted_data[:, dx:] = constant
    elif dx > 0:
        shifted_data[:, 0:dx] = constant

    shifted_data = np.roll(shifted_data, dy, axis=0)
    if dy < 0:
        shifted_data[dy:, :] = constant
    elif dy > 0:
        shifted_data[0:dy, :] = constant
    return shifted_data


def get_boundaries(disp, th=1., dilation=10):
    edges_y = np.logical_or(np.pad(np.abs(disp[1:, :] - disp[:-1, :]) > th, ((1, 0), (0, 0))),
                            np.pad(np.abs(disp[:-1, :] - disp[1:, :]) > th, ((0, 1), (0, 0))))
    edges_x = np.logical_or(np.pad(np.abs(disp[:, 1:] - disp[:, :-1]) > th, ((0, 0), (1, 0))),
                            np.pad(np.abs(disp[:, :-1] - disp[:,1:]) > th, ((0, 0), (0, 1))))
    edges = np.logical_or(edges_y,  edges_x).astype(np.float32)

    if dilation > 0:
        kernel = np.ones((dilation, dilation), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

    return edges


# Utilities to load and save weights

def load_ckp(checkpoint_path, cuda, model, optimizer):
    checkpoint = torch.load(checkpoint_path, map_location=cuda)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr = checkpoint['learning_rate']
    return model, optimizer, checkpoint['epoch'], lr


def save_ckp(state, checkpoint_path):
    torch.save(state, checkpoint_path)


def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


# Utilities to read, resize and pad images

def gt_loader(path):
    gt=None
    try:
        if path.endswith("pfm"):
            gt = np.expand_dims(readPFM(path),0)
        if path.endswith("png"):
            gt = np.expand_dims(cv2.imread(path, -1), 0)/256.
        if path.endswith("npy"):
            gt = np.expand_dims(np.load(path, mmap_mode='c'), 0)
    except:
        print('Cannot open groundtruth depth: '+ path)

    # Remove invalid values
    gt[np.isinf(gt)] = 0

    return gt.transpose(1,2,0)


def img_loader(path, mode="passive", height=2160, width=3840):
    img=None
    try:
        if path.endswith("raw"):
            img = np.fromfile(open(path, 'rb'), dtype=np.uint8).reshape(height, width, 3 ) if mode=="passive" else \
                  np.fromfile(open(path, 'rb'), dtype=np.uint8).reshape(height, width, 1)
        else:
            img = cv2.imread(path, -1)
    except:
        print('Cannot open input image: '+ path)
    return img


def readPFM(file):
    file = open(file, 'rb')
    header = file.readline().rstrip()
    if (sys.version[0]) == '3':
        header = header.decode('utf-8')
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    if (sys.version[0]) == '3':
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    else:
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    if (sys.version[0]) == '3':
        scale = float(file.readline().rstrip().decode('utf-8'))
    else:
        scale = float(file.readline().rstrip())

    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data


def pad_imgs(left, right, height=1024, width=1024, divisor=64):
    top_pad = 0 if (height % divisor) == 0 else  divisor - (height % divisor)
    left_pad = 0 if (width % divisor) == 0 else divisor - (width % divisor)
    left = np.lib.pad(left, ((0, 0), (top_pad, 0), (0, left_pad)), mode='edge')
    right = np.lib.pad(right, ((0, 0), (top_pad, 0), (0, left_pad)), mode='edge')
    return left, right


def pad_img(img, height=1024, width=1024, divisor=64):
    top_pad = 0 if (height % divisor) == 0 else  divisor - (height % divisor)
    left_pad = 0 if (width % divisor) == 0 else divisor - (width % divisor)
    img = np.lib.pad(img, ((0, 0), (top_pad, 0), (0, left_pad)), mode='edge')
    return img


def depad_img(img, height=1024, width=1024, scale_factor=-1, divisor=64):
    top_pad = 0 if (height % divisor) == 0 else  (divisor - (height % divisor))
    left_pad = 0 if (width % divisor) == 0 else (divisor - width % divisor)

    if scale_factor > 0:
        top_pad = int(top_pad * scale_factor)
        left_pad = int(left_pad * scale_factor)

    if img.ndim > 2:
        return img[:, top_pad:, :-left_pad] if left_pad > 0 else \
               img[:, top_pad:, :]
    else:
        return img[top_pad:, :-left_pad] if left_pad > 0 else \
               img[top_pad:, :]

def resize_imgs(imgs):
    dim = len(imgs[0].shape)
    if dim==3:
        assert(imgs[0].shape[0]==3)
        height = imgs[0].shape[1]
        width = imgs[0].shape[2]
    elif dim==2:
        height = imgs[0].shape[0]
        width = imgs[0].shape[1]
    else:
        raise RuntimeError("Unsupported dimension!")

    for idx in range(1, len(imgs)):
        if dim==3:
            imgs[idx] = cv2.resize(imgs[idx].transpose(1,2,0), (width, height)).transpose(2,0,1)
        else:
            imgs[idx] = cv2.resize(imgs[idx], (width, height), interpolation = cv2.INTER_NEAREST)
    return


# Visualization utilities

_color_map_errors = np.array([
    [149, 54, 49],
    [180, 117, 69],
    [209, 173, 116],
    [233, 217, 171],
    [248, 243, 224],
    [144, 224, 254],
    [97, 174, 253],
    [67, 109, 244],
    [39, 48, 215],
    [38, 0, 165],
    [38, 0, 165]
]).astype(float)

_color_map_errors_kitti = np.array([
        [ 0,       0.1875, 149,  54,  49],
        [ 0.1875,  0.375,  180, 117,  69],
        [ 0.375,   0.75,   209, 173, 116],
        [ 0.75,    1.5,    233, 217, 171],
        [ 1.5,     3,      248, 243, 224],
        [ 3,       6,      144, 224, 254],
        [ 6,      12,       97, 174, 253],
        [12,      24,       67, 109, 244],
        [24,      48,       39,  48, 215],
        [48,  np.inf,       38,   0, 165]
]).astype(float)

_color_map_depths = np.array([
    [0, 0, 0],
    [0, 0, 255],
    [255, 0, 0],
    [255, 0, 255],
    [0, 255, 0],
    [0, 255, 255],
    [255, 255, 0],
    [255, 255, 255],
    [255, 255, 255],
]).astype(float)

_color_map_bincenters = np.array([
    0.0,
    0.114,
    0.299,
    0.413,
    0.587,
    0.701,
    0.886,
    1.000,
    2.000,
])

def color_error_image(errors, scale=1, mask=None, BGR=True):
    errors_flat = errors.flatten()
    errors_color_indices = np.clip(np.log2(errors_flat / scale + 1e-5) + 5, 0, 9)
    i0 = np.floor(errors_color_indices).astype(int)
    f1 = errors_color_indices - i0.astype(float)
    colored_errors_flat = _color_map_errors[i0, :] * (1 - f1).reshape(-1, 1) + _color_map_errors[i0 + 1,:] * f1.reshape(-1, 1)

    if mask is not None:
        colored_errors_flat[mask.flatten() == 0] = 255
    if not BGR:
        colored_errors_flat = colored_errors_flat[:, [2, 1, 0]]

    return colored_errors_flat.reshape(errors.shape[0], errors.shape[1], 3).astype(np.int)

def color_error_image_kitti(errors, scale=1, mask=None, BGR=True, dilation=1):
    errors_flat = errors.flatten()
    colored_errors_flat = np.zeros((errors_flat.shape[0], 3))
    for col in _color_map_errors_kitti:
        col_mask = np.logical_and(errors_flat>=col[0]/scale, errors_flat<=col[1]/scale)
        colored_errors_flat[col_mask] = col[2:]
        
    if mask is not None:
        colored_errors_flat[mask.flatten() == 0] = 0

    if not BGR:
        colored_errors_flat = colored_errors_flat[:, [2, 1, 0]]

    colored_errors = colored_errors_flat.reshape(errors.shape[0], errors.shape[1], 3).astype(np.int)

    if dilation>0:
        kernel = np.ones((dilation, dilation), np.uint8)
        colored_errors = cv2.dilate(colored_errors, kernel, iterations=1)
    return colored_errors

def color_depth_map(depths, scale=None):
    if scale is None:
        scale = depths.max()

    values = np.clip(depths.flatten() / scale, 0, 1)
    lower_bin = ((values.reshape(-1, 1) >= _color_map_bincenters.reshape(1, -1)) * np.arange(0, 9)).max(axis=1)
    lower_bin_value = _color_map_bincenters[lower_bin]
    higher_bin_value = _color_map_bincenters[lower_bin + 1]
    alphas = (values - lower_bin_value) / (higher_bin_value - lower_bin_value)
    colors = _color_map_depths[lower_bin] * (1 - alphas).reshape(-1, 1) + _color_map_depths[lower_bin + 1] * alphas.reshape(-1, 1)
    return colors.reshape(depths.shape[0], depths.shape[1], 3).astype(np.uint8)
