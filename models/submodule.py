
import math
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.backends import cuda
from chainer import Variable
from chainer import Sequential
from chainer import initializers as ini

from functools import partial


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return Sequential(
        L.Convolution2D(in_planes, out_planes,
                        ksize=kernel_size, stride=stride,
                        pad=dilation if dilation > 1 else pad,
                        dilate=dilation, nobias=True,
                        initialW=ini.Normal(math.sqrt(2. / (kernel_size * kernel_size * out_planes)))),
        L.BatchNormalization(out_planes, eps=1e-5, decay=0.95,
                             initial_gamma=ini.One(), initial_beta=ini.Zero()),
    )


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):
    return Sequential(
        L.ConvolutionND(3, in_planes, out_planes,
                        ksize=kernel_size, stride=stride,
                        pad=pad, nobias=True,
                        initialW=ini.Normal(math.sqrt(2. / (kernel_size * kernel_size * kernel_size * out_planes)))),
        L.BatchNormalization(out_planes, eps=1e-5, decay=0.95,
                             initial_gamma=ini.One(), initial_beta=ini.Zero()),
    )


class BasicBlock(chainer.Chain):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()
        with self.init_scope():
            self.conv1 = Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                    F.relu)

            self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

            self.downsample = downsample
            self.stride = stride

    def __call__(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class disparityregression(chainer.Chain):
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.disp = Variable(cuda.cupy.reshape(cuda.cupy.array(range(maxdisp), dtype=cuda.cupy.float32),
                                               [1, maxdisp, 1, 1]))

    def __call__(self, x):
        disp = F.tile(self.disp, (x.shape[0], 1, x.shape[2], x.shape[3]))
        out = F.sum(x * disp, 1)
        return out


class feature_extraction(chainer.Chain):
    def __init__(self, gpu):
        super(feature_extraction, self).__init__()
        self.gpu = gpu
        with self.init_scope():
            self.inplanes = 32
            self.firstconv = Sequential(convbn(3, 32, 3, 2, 1, 1),
                                        F.relu,
                                        convbn(32, 32, 3, 1, 1, 1),
                                        F.relu,
                                        convbn(32, 32, 3, 1, 1, 1),
                                        F.relu).to_gpu(self.gpu)

            self.layer1 = self._make_layer(
                BasicBlock, 32, 3, 1, 1, 1).to_gpu(self.gpu)
            self.layer2 = self._make_layer(
                BasicBlock, 64, 16, 2, 1, 1).to_gpu(self.gpu)
            self.layer3 = self._make_layer(
                BasicBlock, 128, 3, 1, 1, 1).to_gpu(self.gpu)
            self.layer4 = self._make_layer(
                BasicBlock, 128, 3, 1, 1, 2).to_gpu(self.gpu)

            self.branch1 = Sequential(partial(F.average_pooling_2d, ksize=(64, 64), stride=(64, 64)),
                                      convbn(128, 32, 1, 1, 0, 1),
                                      F.relu).to_gpu(self.gpu)

            self.branch2 = Sequential(partial(F.average_pooling_2d, ksize=(32, 32), stride=(32, 32)),
                                      convbn(128, 32, 1, 1, 0, 1),
                                      F.relu).to_gpu(self.gpu)

            self.branch3 = Sequential(partial(F.average_pooling_2d, ksize=(16, 16), stride=(16, 16)),
                                      convbn(128, 32, 1, 1, 0, 1),
                                      F.relu).to_gpu(self.gpu)

            self.branch4 = Sequential(partial(F.average_pooling_2d, ksize=(8, 8), stride=(8, 8)),
                                      convbn(128, 32, 1, 1, 0, 1),
                                      F.relu).to_gpu(self.gpu)

            self.lastconv = Sequential(convbn(320, 128, 3, 1, 1, 1),
                                       F.relu,
                                       L.Convolution2D(128, 32, ksize=1, stride=1,
                                                       pad=0, nobias=True,
                                                       initialW=ini.Normal(math.sqrt(2. / 32)))).to_gpu(self.gpu)

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                L.Convolution2D(self.inplanes, planes * block.expansion,
                                ksize=1, stride=stride, nobias=True,
                                initialW=ini.Normal(math.sqrt(2. / (planes * block.expansion)))),
                L.BatchNormalization(planes * block.expansion,
                                     eps=1e-5, decay=0.95,
                                     initial_gamma=ini.One(), initial_beta=ini.Zero()),
            )

        layers = Sequential()
        layers.append(block(self.inplanes, planes,
                            stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return layers

    def __call__(self, x):
        
        x = F.copy(x, self.gpu)

        output = self.firstconv(x)
        output = self.layer1(output)
        output_raw = self.layer2(output)
        output = self.layer3(output_raw)
        output_skip = self.layer4(output)


        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.resize_images(
            output_branch1,
            (output_skip.shape[2], output_skip.shape[3])
        )

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.resize_images(
            output_branch2,
            (output_skip.shape[2], output_skip.shape[3])
        )

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.resize_images(
            output_branch3,
            (output_skip.shape[2], output_skip.shape[3])
        )

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.resize_images(
            output_branch4,
            (output_skip.shape[2], output_skip.shape[3])
        )

        output_feature = F.concat((output_raw, output_skip, output_branch4,
                                   output_branch3, output_branch2, output_branch1), axis=1)
        output_feature = self.lastconv(output_feature)

        return output_feature
