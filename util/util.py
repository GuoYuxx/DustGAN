from __future__ import absolute_import, division, print_function

import os
import cv2
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import torch
import numpy as np
from PIL import Image
import scipy.ndimage as ndimage
import os
from util.niqe import niqe
from util.piqe import piqe
import cv2
import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
import os, time
import ntpath

import numpy as np
import scipy.io as sio


import torch.nn.functional as F

def disp_to_depth(disp, min_depth, max_depth):
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def generate(I):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder_path = os.path.join(
        r'.\mono+stereo_odom_640x192\encoder.pth')
    depth_decoder_path = os.path.join(
        r'.\mono+stereo_odom_640x192\depth.pth')

    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)
    depth_decoder.to(device)
    depth_decoder.eval()

    with torch.no_grad():
        input_image = I.to(device)
        features = encoder(input_image)
        outputs = depth_decoder(features)
        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(disp, input_image.shape[2:], mode="bilinear", align_corners=False)

        disp_resized_np = disp_resized.squeeze().cpu().numpy()
        vmax = np.percentile(disp_resized_np, 95)
        normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
        im = pil.fromarray(colormapped_im)

        depth_np = np.array(im)
        depth_cv = cv2.cvtColor(depth_np, cv2.COLOR_RGB2BGR)
        depth_cv = cv2.resize(depth_cv, (input_image.shape[3], input_image.shape[2]))
        depth_float = depth_cv.astype(np.float32)

        min_val = np.min(depth_float)
        max_val = np.max(depth_float)
        normalized_depth = (depth_float - min_val) / (max_val - min_val)
        normalized_depth = normalized_depth * 0.2 + 0.4

        org_image = input_image.squeeze().permute(1, 2, 0).cpu().numpy()
        org_image = cv2.resize(org_image, (input_image.shape[3], input_image.shape[2]))

        # 确保 color_image 使用 RGB 格式
        color_image = cv2.imread(r'.\color\c1.png')
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        color_image = cv2.resize(color_image, (input_image.shape[3], input_image.shape[2]))

        result = 0.8 * (0.8 * org_image * normalized_depth + 1.3 * color_image * (1 - normalized_depth))
        dust_J = result

        # Ensure dust_J has the shape (C, H, W)
        if dust_J.ndim == 3:
            dust_J = np.transpose(dust_J, (2, 0, 1))

        print(f"dust_J min value: {dust_J.min()}, max value: {dust_J.max()}, mean value: {dust_J.mean()}")
        print(f"dust_J shape before resizing: {dust_J.shape}")

        # If resizing is necessary
        if dust_J.shape[1] != input_image.shape[2] or dust_J.shape[2] != input_image.shape[3]:
            dust_J = cv2.resize(dust_J, (input_image.shape[3], input_image.shape[2]))
            print(f"dust_J shape after resizing: {dust_J.shape}")

        return dust_J

def resize_to_minimum_size(image, min_height, min_width):
    current_height, current_width = image.shape[1], image.shape[2]
    if current_height < min_height or current_width < min_width:
        new_height = max(current_height, min_height)
        new_width = max(current_width, min_width)
        image = cv2.resize(image.transpose(1, 2, 0), (new_width, new_height)).transpose(2, 0, 1)
    return image


#融合
def fuse_images(dust_J, refine_J):
    if dust_J.ndim == 2:
        dust_J_rgb = cv2.cvtColor(dust_J, cv2.COLOR_GRAY2RGB)
    else:
        dust_J_rgb = dust_J

    if refine_J.ndim == 2:
        refine_J_rgb = cv2.cvtColor(refine_J, cv2.COLOR_GRAY2RGB)
    else:
        refine_J_rgb = refine_J

    score_recpiqe = piqe(dust_J_rgb)
    score_refinepiqe = piqe(refine_J_rgb)
    score_recniqe = niqe(dust_J_rgb)
    score_refineniqe = niqe(refine_J_rgb)

    fuseWeightniqe = score_recniqe / (score_recniqe + score_refineniqe)
    fuseWeightpiqe = 1 - score_recpiqe / (score_recpiqe + score_refinepiqe)
    fuseWeight = (fuseWeightpiqe + fuseWeightniqe) / 2

    return dust_J * fuseWeight + refine_J * (1 - fuseWeight)

def get_tensor_dark_channel(img, neighborhood_size):
    shape = img.shape
    if len(shape) == 4:
        img_min = torch.min(img, dim=1)
        img_dark = F.max_pool2d(img_min, kernel_size=neighborhood_size, stride=1)
    else:
        raise NotImplementedError('get_tensor_dark_channel is only for 4-d tensor [N*C*H*W]')

    return img_dark



def array2Tensor(in_array, gpu_id=-1):
    in_shape = in_array.shape
    if len(in_shape) == 2:
        in_array = in_array[:,:,np.newaxis]

    arr_tmp = in_array.transpose([2,0,1])
    arr_tmp = arr_tmp[np.newaxis,:]

    if gpu_id >= 0:
        return torch.tensor(arr_tmp.astype(np.float)).to(gpu_id)
    else:
        return torch.tensor(arr_tmp.astype(np.float))


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def rescale_tensor(input_tensor):
    """"Converts a Tensor array into the Tensor array whose data are identical to the image's.
    [height, width] not [width, height]

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """

    if isinstance(input_tensor, torch.Tensor):
        input_tmp = input_tensor.cpu().float()
        output_tmp = input_tmp * 255.0
        output_tmp = output_tmp.to(torch.uint8)
    else:
        return input_tensor

    return output_tmp.to(torch.float32) / 255.0

    # if not isinstance(input_image, np.ndarray):
    #     if isinstance(input_image, torch.Tensor):  # get the data from a variable
    #         image_tensor = input_image.data
    #     else:
    #         return input_image
    #     image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
    #     image_numpy = (image_numpy + 1) / 2.0 * white_color  # post-processing: tranpose and scaling
    # else:  # if it is a numpy array, do nothing
    #     image_numpy = input_image
    # return torch.from_numpy(image_numpy)

def my_imresize(in_array, tar_size):
    oh = in_array.shape[0]
    ow = in_array.shape[1]

    if len(tar_size) == 2:
        h_ratio = tar_size[0]/oh
        w_ratio = tar_size[1]/ow
    elif len(tar_size) == 1:
        h_ratio = tar_size
        w_ratio = tar_size

    if len(in_array.shape) == 3:
        return ndimage.zoom(in_array, (h_ratio, w_ratio, 1), prefilter=False)
    else:
        return ndimage.zoom(in_array, (h_ratio, w_ratio), prefilter=False)

def psnr(img, ref, max_val=1):
    if isinstance(img, torch.Tensor):
        distImg = img.cpu().float().numpy()
    elif isinstance(img, np.ndarray):
        distImg = img.astype(np.float)
    else:
        distImg = np.array(img).astype(np.float)

    if isinstance(ref, torch.Tensor):
        refImg = ref.cpu().float().numpy()
    elif isinstance(ref, np.ndarray):
        refImg = ref.astype(np.float)
    else:
        refImg = np.array(ref).astype(np.float)

    rmse = np.sqrt( ((distImg-refImg)**2).mean() )
    # rmse = np.std(distImg-refImg) # keep the same with RESIDE's criterion
    return 20*np.log10(max_val/rmse)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    print(f"Saving image to {image_path}")
    print(f"Image stats before saving - min value: {image_numpy.min()}, max value: {image_numpy.max()}, mean value: {image_numpy.mean()}")

    if image_numpy.ndim == 2:
        image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_GRAY2RGB)
    elif image_numpy.shape[0] == 1:
        image_numpy = np.repeat(image_numpy, 3, axis=0)
    elif image_numpy.shape[0] == 3:
        image_numpy = image_numpy.transpose(1, 2, 0)

    if image_numpy.dtype != np.uint8:
        image_numpy = (image_numpy * 255).astype(np.uint8)

    print(f"Image stats after converting to uint8 - min value: {image_numpy.min()}, max value: {image_numpy.max()}, mean value: {image_numpy.mean()}")

    image_pil = pil.fromarray(image_numpy)
    image_pil.save(image_path)



def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
