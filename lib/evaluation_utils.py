import torch
import math
import matplotlib.pyplot as plt
from .utils import *
from tqdm import tqdm
import os
import time

def soft_edge_error(pred, gt, radius=1):
    abs_diff=[]
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            abs_diff.append(np.abs(shift_2d_replace(gt, i, j, 0) - pred))
    return np.minimum.reduce(abs_diff)

def compute_bad(disp_diff, mask, th=3):
    bad_pixels =  disp_diff > th
    return 100.0 * bad_pixels.sum() / mask.sum()

def eval_disp(pred, gt, mask):
    epe = 0
    bad_1 = 0
    bad_2 = 0
    bad_3 = 0

    if mask.sum() > 0:
        disp_diff = np.abs(gt[mask] - pred[mask])
        epe = disp_diff.mean()
        bad_1 = compute_bad(disp_diff, mask, 1.)
        bad_2 = compute_bad(disp_diff, mask, 2.)
        bad_3 = compute_bad(disp_diff, mask, 3.)

    return epe, bad_1, bad_2, bad_3

def eval_edges(pred, gt, mask, th=1., dilation=0):
    see_disp = 0
    edges = get_boundaries(gt, th=th, dilation=dilation)
    mask = np.logical_and(mask, edges)
    if mask.sum() > 0:
        see_disp = soft_edge_error(pred, gt)[mask].mean()
    return see_disp

def inference(net, cuda, height=1536, width=2048, num_samples=200000, num_out=1):
    nx = np.linspace(0, width - 1, width)
    ny = np.linspace(0, height - 1, height)
    u, v = np.meshgrid(nx, ny)

    coords = np.expand_dims(np.stack((u.flatten(), v.flatten()), axis=-1), 0)
    batch_size, n_pts, _= coords.shape
    coords = torch.Tensor(coords).float().to(device=cuda)
    output = torch.zeros(num_out, math.ceil(width * height / num_samples), num_samples)

    with torch.no_grad():
        for i, p_split in enumerate(torch.split(coords.reshape(batch_size, -1, 2), int(num_samples / batch_size), dim=1)):
            points = torch.transpose(p_split, 1, 2)
            net.query(points.to(device=cuda))
            preds = net.get_preds()
            for k in range(num_out):
                output[k, i, :p_split.shape[1]] = preds[k].to(device=cuda)
    res = []
    for i in range(num_out):
        res.append(output[i].view( 1, -1)[:,:n_pts].reshape(-1, height, width))
    return res

def predict(net, cuda, data, superes_factor=1., compute_entropy=False):
    output = {}
    left = data['left'].to(device=cuda)
    right = data['right'].to(device=cuda)
    shape = data['o_shape'].to(device=cuda)

    net.filter(left, right, phase='test')

    num_out = {"standard": 1,
               "unimodal": 2,
               "bimodal": 7}
    try:
        height = left[0].shape[1]
        width = left[0].shape[2]

        h = to_numpy(shape[1])
        w = to_numpy(shape[0])
        s = superes_factor

        res = inference(net, cuda, height=height * superes_factor,
                                   width=width * superes_factor,
                                   num_out=num_out[net.output_representation])

        output["pred_disp"] = depad_img(to_numpy(res[0]), h, w, s)

        if net.output_representation == 'bimodal':
            output["mu0"] = depad_img(to_numpy(res[1]), h, w, s)
            output["mu1"] = depad_img(to_numpy(res[2]), h, w, s)
            output["sigma0"] = depad_img(to_numpy(res[3]), h, w, s)
            output["sigma1"] = depad_img(to_numpy(res[4]), h, w, s)
            output["pi0"] = depad_img(to_numpy(res[5]), h, w, s)
            output["pi1"] = depad_img(to_numpy(res[6]), h, w, s)
            if compute_entropy:
                output["uncertainty"] = differential_entropy(output["mu0"]/net.maxdisp, output["mu1"]/net.maxdisp,
                                                             output["sigma0"], output["sigma1"],
                                                             output["pi0"], output["pi1"])
        elif net.output_representation == 'unimodal':
            output["var"] = depad_img(to_numpy(res[1]))

        return output

    except Exception as e:
        print(e)
        print('Can not generate the disparity map at this time.')

def validation(opt, net, cuda, data, num_gen_test=10, num_write_test=5, save_imgs=False):
    metrics = {}
    left_imgs = []
    preds = []
    gts = []

    SEE = np.zeros(num_gen_test, np.float32)
    EPE = np.zeros(num_gen_test, np.float32)
    bad1_all = np.zeros(num_gen_test, np.float32)
    bad2_all = np.zeros(num_gen_test, np.float32)
    bad3_all = np.zeros(num_gen_test, np.float32)

    valid_pixels = 0
    total_pixels = 0
    avg_time = 0

    write_freq = num_gen_test // num_write_test
    write_freq = max(write_freq, 1)

    print(" * Evaluation")
    for gen_idx in tqdm(range(num_gen_test)):
        test_data = data[gen_idx]

        # Inference
        start = time.clock()
        output = predict(net, cuda, test_data, opt.superes_factor, opt.differential_entropy)
        elapsed = time.clock() - start

        disp = output['pred_disp']
        gt_disp = to_numpy(test_data['gt_disp'])
        left_img = depad_img(to_numpy(test_data['left']), disp.shape[0], disp.shape[1])

        # Evaluation
        mask = np.logical_and(gt_disp > opt.mindisp, gt_disp < opt.maxdisp)
        epe, bad_1, bad_2, bad_3 = eval_disp(disp, gt_disp, mask)
        see = eval_edges(disp, gt_disp, mask)

        if gen_idx % write_freq == 0:
            preds.append(disp)
            gts.append(gt_disp)
            
            if opt.mode == "passive":
                left_imgs.append(left_img)
            else:
                left_imgs.append(np.repeat(np.expand_dims(left_img, 0), 3, 0))

        if save_imgs:
            # Save images
            save_path = '%s/%s/%s' % (opt.results_path, test_data['name_scene'], test_data['name'])
            os.makedirs(save_path, exist_ok=True)
            for key, value in output.items():
                #np.save(os.path.join(save_path, '%s.npy' % key), value)
                plt.imsave(os.path.join(save_path, '%s.png' % key), value, cmap='jet')
            plt.imsave(os.path.join(save_path, 'pred_disp_kitti_color.jpg'), color_depth_map(disp)/255.)
            plt.imsave(os.path.join(save_path, 'EPE.png'), color_error_image_kitti(np.abs(disp - gt_disp), mask=mask, BGR=False)/255.)
            
        SEE[gen_idx] = see
        EPE[gen_idx] = epe
        bad1_all[gen_idx] = bad_1
        bad2_all[gen_idx] = bad_2
        bad3_all[gen_idx] = bad_3
        valid_pixels = valid_pixels + mask.sum()
        total_pixels = total_pixels + np.ones_like(mask).sum()
        avg_time += elapsed

    metrics["SEE"] = SEE.mean()
    metrics["EPE"] = EPE.mean()
    metrics["bad1"] = bad1_all.mean()
    metrics["bad2"] = bad2_all.mean()
    metrics["bad3"] = bad3_all.mean()

    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(
          'see_disp', 'EPE', 'bad_1', 'bad_2', 'bad_3', 'density', 'avg_time [s]'))

    print("{:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:>10.3f}, {:>10.3f}".format(
           metrics["SEE"], metrics["EPE"], metrics["bad1"], metrics["bad2"], metrics["bad3"], 
           100 * valid_pixels / total_pixels, avg_time / num_gen_test))

    # resize images (KITTI val images have difference size) - for visualization only
    resize_imgs(left_imgs)
    resize_imgs(preds)
    resize_imgs(gts)

    return metrics, left_imgs, preds, gts
