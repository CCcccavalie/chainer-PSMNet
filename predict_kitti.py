#!/usr/bin/env python3
import numpy as np
import argparse
import matplotlib
import cv2
matplotlib.use('Agg')

import chainer
from chainer import training
from chainer.training import extensions
from chainer.training import triggers
from chainer.backends import cuda
import chainer.functions as F
from chainer import serializers

# library
#from dataloader_SceneFlow import listflowfile as lt
#from dataloader_SceneFlow import SceneFlowLoader as DA

from dataloader_KITTI import KITTIloader2015 as lt
from dataloader_KITTI import KITTILoader as DA
from models import *
from extension import *


def main():
    chainer.config.train = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--maxdisp', type=int, default=192,
                        help='maxium disparity')
    parser.add_argument('--gpu0', '-g', type=int, default=-1,
                        help='First GPU ID (negative value indicates CPU)')
    parser.add_argument('--gpu1', '-G', type=int, default=-1,
                        help='Second GPU ID (negative value indicates CPU)')
    parser.add_argument('--datapath', default='/home/<username>/datasets/KITTI_stereo/training/',
                        help='datapath')
    parser.add_argument('--model', default='/<modelpath>/model_iter_xxx.npz',
                        help='datapath')
    args = parser.parse_args()

    print('# GPU: {} ({})'.format(args.gpu0, args.gpu1))
    print('# datapath: {}'.format(args.datapath))
    print('')

    # Model
    model = basic(args.maxdisp, args.gpu0, args.gpu1,
                  train_type=None, training=False)

    # load model
    serializers.load_npz(args.model, model)

    if args.gpu0 >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu0).use()

    # Dataset
    # dataloader
    dataname_list = lt.dataloader(args.datapath)
    all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = dataname_list
    # transform
    train = DA.myImageFolder(all_left_img, all_right_img, all_left_disp, True)
    test = DA.myImageFolder(
        test_left_img, test_right_img, test_left_disp, False)

    data_num = 0
    for data_num in range(5):
        train_left, train_right, train_disp = test.get_example(data_num)
        # to Variable
        train_left = F.copy(chainer.Variable(cuda.cupy.expand_dims(train_left, 0)),args.gpu0)
        train_right = F.copy(chainer.Variable(cuda.cupy.expand_dims(train_right, 0)),args.gpu0)
        train_disp = F.copy(chainer.Variable(cuda.cupy.expand_dims(train_disp, 0)),args.gpu0)

        with chainer.using_config('train', False):    
            
            pred = model(train_left, train_right, train_disp)[0].transpose(1,2,0)

            train_left = cuda.to_cpu(train_left[0].array.transpose(1,2,0))[:,:,::-1]
            train_right = cuda.to_cpu(train_right[0].array.transpose(1,2,0))[:,:,::-1]
            train_disp = cuda.to_cpu(train_disp.array.transpose(1,2,0))
            pred = cuda.to_cpu(pred.array)
            pred = (pred * 255 / pred.max()).astype(np.uint8)
            pred_rainbow = cv2.applyColorMap(pred, cv2.COLORMAP_RAINBOW)

            cv2.imwrite("test{}_0.png".format(data_num),train_left*255./train_left.max())
            cv2.imwrite("test{}_1.png".format(data_num),train_right*255./train_right.max())
            cv2.imwrite("test{}_2.png".format(data_num),train_disp*255./train_disp.max())
            cv2.imwrite("test{}_3.png".format(data_num),pred_rainbow)


if __name__ == "__main__":
    main()
