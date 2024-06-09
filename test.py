from __future__ import absolute_import, division, print_function


import cv2
import os, time
import ntpath

import numpy as np


from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import util

def test():
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    if opt.save_image:
        curSaveFolder = os.path.join(opt.dataroot, opt.method_name)
        if not os.path.exists(curSaveFolder):
            os.makedirs(curSaveFolder, mode=0o777)

    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()

    time_total = 0
    for i, data in enumerate(dataset):
        # if i <= 627:
        #     continue

        img_path = data['paths']
        short_path = ntpath.basename(img_path[0])
        name = os.path.splitext(short_path)[0]
        print('%s [%d]' % (short_path, i + 1))
        # print(data['B_paths'])

        # if i >= opt.num_test:  # only apply our model to opt.num_test images.
        #     break
        t0 = time.time()
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        time_total += time.time() - t0

        visuals = model.get_current_visuals()

        refine_J = util.tensor2im(visuals['refine_J'], np.float64) / 255.  # [0, 1]
        dust_J = util.tensor2im(visuals['dust_J'], np.float64)  # [0, 255] 的浮点数
        print(f"dust_J shape before resizing: {dust_J.shape}")

        dust_J = util.resize_to_minimum_size(dust_J, 192, 192)
        print(f"dust_J shape after resizing: {dust_J.shape}")

        dust_J = np.clip(dust_J, 0, 1)
        dust_J_scaled = dust_J * 255
        dust_J_uint8 = dust_J_scaled.astype(np.uint8)
        dust_J_gray = cv2.cvtColor(dust_J_uint8.transpose(1, 2, 0),
                                   cv2.COLOR_RGB2GRAY) if dust_J_uint8.ndim == 3 else dust_J_uint8
        dust_J_rgb = cv2.cvtColor(dust_J_gray, cv2.COLOR_GRAY2RGB)

        real_I = util.tensor2im(data['dusty'], np.float64)  # [0, 255], np.float
        fused_J = util.fuse_images(dust_J_rgb, refine_J * 255.) / 255.
        fusedImg = (fused_J * 255).astype(np.uint8)

        util.save_image(fusedImg, os.path.join(curSaveFolder, f'{name}_fuse.png'))
        util.save_image((refine_J * 255).astype(np.uint8), os.path.join(curSaveFolder, f'{name}_refine.png'))

        if dust_J.ndim == 3 and dust_J.shape[0] == 3:
            dust_J = dust_J.transpose(1, 2, 0)  # 从 (C, H, W) 转换到 (H, W, C)
        elif dust_J.ndim == 2:
            dust_J = cv2.cvtColor(dust_J, cv2.COLOR_GRAY2RGB)
        elif dust_J.shape[0] == 1:
            dust_J = np.repeat(dust_J, 3, axis=0).transpose(1, 2, 0)

        print(
            f"Image stats before saving - min value: {dust_J.min()}, max value: {dust_J.max()}, mean value: {dust_J.mean()}")
        dust_J = np.clip(dust_J * 255, 0, 255).astype(np.uint8)


        util.save_image(dust_J.astype(np.uint8), os.path.join(curSaveFolder, f'{name}_dust.png'))
        saved_dust_J = cv2.imread(os.path.join(curSaveFolder, f'{name}_dust.png'))
        print(
            f"Image stats after saving - min value: {saved_dust_J.min()}, max value: {saved_dust_J.max()}, mean value: {saved_dust_J.mean()}")

        print(f"num: {len(dataset)}")
        print(f"average time: {time_total / len(dataset)}")


if __name__ == '__main__':
    test()



