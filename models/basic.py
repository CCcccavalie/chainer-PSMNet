#!/usr/bin/env python3
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.backends import cuda
from chainer import Variable
from chainer import initializers
from chainer import Sequential
from chainer import initializers as ini

from .submodule import *


class PSMNet(chainer.Chain):
    def __init__(self, maxdisp, gpu0, gpu1, training=True, train_type=None):
        super(PSMNet, self).__init__()
        self.gpu0 = gpu0
        self.gpu1 = gpu1
        self.training = training
        self.train_type = train_type

        with self.init_scope():
            self.maxdisp = maxdisp
            self.feature_extraction = feature_extraction(self.gpu0)

            self.dres0 = Sequential(convbn_3d(64, 32, 3, 1, 1),
                                    F.relu,
                                    convbn_3d(32, 32, 3, 1, 1),
                                    F.relu).to_gpu(self.gpu1)

            self.dres1 = Sequential(convbn_3d(32, 32, 3, 1, 1),
                                    F.relu,
                                    convbn_3d(32, 32, 3, 1, 1)).to_gpu(self.gpu1)

            self.dres2 = Sequential(convbn_3d(32, 32, 3, 1, 1),
                                    F.relu,
                                    convbn_3d(32, 32, 3, 1, 1)).to_gpu(self.gpu1)

            self.dres3 = Sequential(convbn_3d(32, 32, 3, 1, 1),
                                    F.relu,
                                    convbn_3d(32, 32, 3, 1, 1)).to_gpu(self.gpu1)

            self.dres4 = Sequential(convbn_3d(32, 32, 3, 1, 1),
                                    F.relu,
                                    convbn_3d(32, 32, 3, 1, 1)).to_gpu(self.gpu1)

            self.classify = Sequential(convbn_3d(32, 32, 3, 1, 1),
                                       F.relu,
                                       L.ConvolutionND(3, 32, 1, ksize=3, stride=1, pad=1, nobias=True,
                                                       initialW=ini.Normal(math.sqrt(2. / (3 * 3 * 3 * 1))))).to_gpu(self.gpu1)

    def __call__(self, left, right, disp_true):

        refimg_fea = self.feature_extraction(left)
        targetimg_fea = self.feature_extraction(right)
        # matching
        # with chainer.no_backprop_mode():
        cost = None

        for i in range(int(self.maxdisp / 4)):
            if i > 0:
                # limit size i
                cost_i = F.concat((refimg_fea[:, :, :, i:],
                                   targetimg_fea[:, :, :, :-i]),
                                  axis=1).reshape(refimg_fea.shape[0],
                                                  refimg_fea.shape[1] * 2,
                                                  1,
                                                  refimg_fea.shape[2],
                                                  refimg_fea.shape[3] - i)
                cost_zero = Variable(cuda.cupy.zeros((refimg_fea.shape[0],
                                                      int(refimg_fea.shape[1] * 2),
                                                      1,
                                                      refimg_fea.shape[2],
                                                      i), dtype=cuda.cupy.float32))
                cost_i = F.concat((cost_zero, cost_i), axis=4)
                cost = F.concat((cost, cost_i), axis=2)
            else:
                cost = F.concat((refimg_fea, targetimg_fea),
                                axis=1).reshape(refimg_fea.shape[0],
                                                refimg_fea.shape[1] * 2,
                                                1,
                                                refimg_fea.shape[2],
                                                refimg_fea.shape[3])

        # gpu0 to gpu1
        cost = F.copy(cost, self.gpu1)

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0
        cost0 = self.dres2(cost0) + cost0
        cost0 = self.dres3(cost0) + cost0
        cost0 = self.dres4(cost0) + cost0
        cost = self.classify(cost0)

        # gpu1 to gpu0
        cost = F.copy(cost, self.gpu0)

        cost = F.unpooling_nd(cost, 4,
                              outsize=(self.maxdisp, left.shape[2], left.shape[3]))
        cost = F.average_pooling_nd(cost, 3, 1, 1)
        # here insert average_pooling_nd(kernel=3, stride=1) for trilinear upsampling !!!
        cost = F.squeeze(cost, 1)
        pred = F.softmax(cost)  # ???
        pred = disparityregression(self.maxdisp)(pred)

        # calculate loss
        pred = F.clip(pred.reshape(pred.shape[0], -1), 0., float(self.maxdisp))
        disp_true = disp_true.reshape(disp_true.shape[0], -1)

        # mask
        if self.train_type == "kitti":
            pred_mask = F.where(disp_true > 0., pred, disp_true)
        elif self.train_type == "sceneflow":
            pred_mask = F.where(disp_true < maxdisp, pred, disp_true)
        else:
            pred_mask = pred

        #mask = Variable(disp_true).array < self.maxdisp
        loss = F.huber_loss(pred_mask, disp_true, delta=1)
        loss = F.average(loss / pred_mask.shape[1])

        chainer.reporter.report({'loss': loss}, self)
        
        if self.training:
            return loss
        else:
            return pred.reshape(1, 1, left.shape[2], right.shape[3])
