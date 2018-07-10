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


class hourglass(chainer.Chain):
    def __init__(self, inplanes, gpu):
        super(hourglass, self).__init__()
        self.gpu = gpu

        self.conv1 = Sequential(convbn_3d(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                F.relu).to_gpu(self.gpu)

        self.conv2 = convbn_3d(inplanes * 2, inplanes * 2,
                               kernel_size=3, stride=1, pad=1).to_gpu(self.gpu)

        self.conv3 = Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                F.relu).to_gpu(self.gpu)

        self.conv4 = Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1),
                                F.relu).to_gpu(self.gpu)

        self.conv5 = Sequential(L.DeconvolutionND(3, inplanes * 2, inplanes * 2, ksize=4, stride=2, pad=1, nobias=True,
                                                  initialW=ini.Normal(math.sqrt(2. / 32))),
                                L.BatchNormalization(inplanes * 2, eps=1e-5, decay=0.95,
                                                     initial_gamma=ini.One(), initial_beta=ini.Zero())
                                ).to_gpu(self.gpu)  # +conv2

        self.conv6 = Sequential(L.DeconvolutionND(3, inplanes * 2, inplanes, ksize=4, stride=2, pad=1, nobias=True),
                                L.BatchNormalization(inplanes, eps=1e-5, decay=0.95,
                                                     initial_gamma=ini.One(), initial_beta=ini.Zero())
                                ).to_gpu(self.gpu)  # +x

    def __call__(self, x, presqu, postsqu):

        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        if postsqu is not None:
            pre = F.relu(pre + postsqu)
        else:
            pre = F.relu(pre)

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu)  # in:1/16 out:1/8
        else:
            post = F.relu(self.conv5(out) + pre)

        out = self.conv6(post)  # in:1/8 out:1/4

        return out, pre, post


class PSMNet(chainer.Chain):
    def __init__(self, maxdisp, gpu0, gpu1, gpu2, training=True, train_type=None):
        super(PSMNet, self).__init__()
        self.gpu0 = gpu0
        self.gpu1 = gpu1
        self.gpu2 = gpu2
        self.training = training
        self.train_type = train_type

        with self.init_scope():
            self.maxdisp = maxdisp
            self.feature_extraction = feature_extraction(self.gpu1)

            self.dres0 = Sequential(convbn_3d(64, 32, 3, 1, 1),
                                    F.relu,
                                    convbn_3d(32, 32, 3, 1, 1),
                                    F.relu).to_gpu(self.gpu2)

            self.dres1 = Sequential(convbn_3d(32, 32, 3, 1, 1),
                                    F.relu,
                                    convbn_3d(32, 32, 3, 1, 1)).to_gpu(self.gpu2)

            self.dres2 = hourglass(32, self.gpu2)

            self.dres3 = hourglass(32, self.gpu2)

            self.dres4 = hourglass(32, self.gpu2)

            self.classify1 = Sequential(convbn_3d(32, 32, 3, 1, 1),
                                        F.relu,
                                        L.ConvolutionND(3, 32, 1, ksize=3, stride=1, pad=1, nobias=True,
                                                        initialW=ini.Normal(math.sqrt(2. / (3 * 3 * 3 * 1))))).to_gpu(self.gpu2)

            self.classify2 = Sequential(convbn_3d(32, 32, 3, 1, 1),
                                        F.relu,
                                        L.ConvolutionND(3, 32, 1, ksize=3, stride=1, pad=1, nobias=True,
                                                        initialW=ini.Normal(math.sqrt(2. / (3 * 3 * 3 * 1))))).to_gpu(self.gpu2)

            self.classify3 = Sequential(convbn_3d(32, 32, 3, 1, 1),
                                        F.relu,
                                        L.ConvolutionND(3, 32, 1, ksize=3, stride=1, pad=1, nobias=True,
                                                        initialW=ini.Normal(math.sqrt(2. / (3 * 3 * 3 * 1))))).to_gpu(self.gpu2)

    def __call__(self, left, right, disp_true):
        # gpu0 to gpu1
        left = F.copy(left, self.gpu1)
        right = F.copy(right, self.gpu1)

        refimg_fea = self.feature_extraction(left)
        targetimg_fea = self.feature_extraction(right)
        
        refimg_fea = F.copy(refimg_fea, self.gpu0)
        targetimg_fea = F.copy(targetimg_fea, self.gpu0)

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

        cost = F.copy(cost, self.gpu2)

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1 + cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2 + cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2)
        out3 = out3 + cost0

        cost1 = self.classify1(out1)
        cost2 = self.classify2(out2) + cost1
        cost3 = self.classify3(out3) + cost2

        # gpu1 to gpu0
        left = F.copy(left, self.gpu0)
        right = F.copy(right, self.gpu0)
        #disp_true = F.copy(disp_true, self.gpu0)
        cost1 = F.copy(cost1, self.gpu0)
        cost2 = F.copy(cost2, self.gpu0)
        cost3 = F.copy(cost3, self.gpu0)

        if self.training:
            # trilinear upsample
            cost1 = F.unpooling_nd(cost1, 4,
                                   outsize=(self.maxdisp, left.shape[2], left.shape[3]))
            cost1 = F.average_pooling_nd(cost1, 3, 1, 1)

            cost2 = F.unpooling_nd(cost2, 4,
                                   outsize=(self.maxdisp, left.shape[2], left.shape[3]))
            cost2 = F.average_pooling_nd(cost2, 3, 1, 1)

            # for cost1
            cost1 = F.squeeze(cost1, 1)
            pred1 = F.softmax(cost1)  # ???
            pred1 = disparityregression(self.maxdisp)(pred1)

            # for cost2
            cost2 = F.squeeze(cost2, 1)
            pred2 = F.softmax(cost2)  # ???
            pred2 = disparityregression(self.maxdisp)(pred2)

        # for cost3
        cost3 = F.unpooling_nd(cost3, 4,
                               outsize=(self.maxdisp, left.shape[2], left.shape[3]))
        cost3 = F.average_pooling_nd(cost3, 3, 1, 1)

        cost3 = F.squeeze(cost3, 1)
        pred3 = F.softmax(cost3)  # ???
        pred3 = disparityregression(self.maxdisp)(pred3)


        def calculate_disp_loss(pred, disp_true, train_type):
            # calculate loss
            pred = F.clip(pred.reshape(
                pred.shape[0], -1), 0., float(self.maxdisp))
            disp_true = disp_true.reshape(disp_true.shape[0], -1)

            # mask
            if train_type == "kitti":
                pred_mask = F.where(disp_true > 0., pred, disp_true)
            elif train_type == "sceneflow":
                pred_mask = F.where(disp_true < float(self.maxdisp), pred, disp_true)
            else:
                pred_mask = pred

            #mask = Variable(disp_true).array < self.maxdisp
            loss = F.huber_loss(pred_mask, disp_true, delta=1)
            loss = F.average(loss / pred_mask.shape[1])
            return loss

        if self.training:
            loss1 = calculate_disp_loss(pred1, disp_true, self.train_type)
            loss2 = calculate_disp_loss(pred2, disp_true, self.train_type)
            loss3 = calculate_disp_loss(pred3, disp_true, self.train_type)
            loss = loss1 + loss2 + loss3

            chainer.reporter.report({'loss1': loss1,
                                    'loss2': loss2,
                                    'loss3': loss3,
                                    'loss': loss}, self)

            return loss
        else:
            return pred3.reshape(1, 1, left.shape[2], right.shape[3])
