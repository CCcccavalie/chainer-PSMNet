#!/usr/bin/env python3
import argparse
import matplotlib
matplotlib.use('Agg')

import chainer
from chainer import training
from chainer.training import extensions
from chainer.training import triggers

# library
from dataloader_SceneFlow import listflowfile as lt
from dataloader_SceneFlow import SceneFlowLoader as DA

#from dataloader_KITTI import KITTIloader2015 as lt
#from dataloader_KITTI import KITTILoader as DA
from models import *


class TestModeEvaluator(extensions.Evaluator):
    def evaluate(self):
        model = self.get_target('main')
        model.train = False
        ret = super(TestModeEvaluator, self).evaluate()
        model.train = True
        return ret


def main():
    parser = argparse.ArgumentParser(description='Chainer-PSMNet')
    parser.add_argument('--maxdisp', type=int, default=192,
                        help='maxium disparity')
    parser.add_argument('--datapath', default='/home/<username>/datasets/SceneFlow/',  # /home/<username>/datasets/KITTI_stereo/training/
                        help='datapath')
    parser.add_argument('--model_type', default='basic',
                        help='model type')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train')
    parser.add_argument('--batchsize', type=int, default=12)
    parser.add_argument('--gpu0', '-g', type=int, default=-1,
                        help='First GPU ID (negative value indicates CPU)')
    parser.add_argument('--gpu1', '-G1', type=int, default=-1,
                        help='Second GPU ID (negative value indicates CPU)')
    parser.add_argument('--gpu2', '-G2', type=int, default=-1,
                        help='Third GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--out', default='result/basic')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()

    print('# GPU: {} ({},{})'.format(args.gpu0, args.gpu1, args.gpu2))
    print('# datapath: {}'.format(args.datapath))
    print('# epoch: {}'.format(args.epochs))
    print('# plot: {}'.format(extensions.PlotReport.available()))
    print('')

    # Triggers
    log_trigger = (3, 'iteration')
    validation_trigger = (1, 'epoch')
    snapshot_trigger = (1, 'epoch')
    end_trigger = (args.epochs, 'epoch')

    # Dataset
    # dataloader
    dataname_list = lt.dataloader(args.datapath)
    all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = dataname_list
    # transform
    train = DA.myImageFolder(all_left_img, all_right_img, all_left_disp, True)
    test = DA.myImageFolder(
        test_left_img, test_right_img, test_left_disp, False)

    # Iterator
    train_iter = chainer.iterators.MultiprocessIterator(train, args.batchsize)
    test_iter = chainer.iterators.MultiprocessIterator(test, args.batchsize,
                                                       shuffle=False, repeat=False)

    # Model
    if args.model_type == "basic":
        model = basic(args.maxdisp, args.gpu0, args.gpu1,
                      training=True, train_type="sceneflow")
    elif args.model_type == "stackhourglass":
        model = stackhourglass(args.maxdisp, args.gpu0, args.gpu1, args.gpu2,
                               training=True, train_type="sceneflow")
    else:
        print("Error : model name error")
        exit()

    if args.gpu0 >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu0).use()
        # model.to_gpu()  # Copy the model to the GPU

    # Optimizer
    optimizer = chainer.optimizers.Adam(alpha=0.001, beta1=0.9, beta2=0.999)
    optimizer.setup(model)
    # optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))

    # Updater
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu0)

    # Trainer
    trainer = training.Trainer(updater, end_trigger, args.out)

    trainer.extend(extensions.LogReport(trigger=log_trigger))
    trainer.extend(extensions.observe_lr(), trigger=log_trigger)
    trainer.extend(extensions.dump_graph('main/loss'))
    # trainer.extend(TestModeEvaluator(test_iter, model, device=args.gpu0),
    #               trigger=log_trigger)

    # plot loss
    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(
            ['main/loss', 'main/loss1', 'main/loss2', 'main/loss3'], x_key='iteration',
            file_name='loss.png'))

    # print progression
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'elapsed_time',
         'lr', 'main/loss', 'validation/main/loss']),
        trigger=log_trigger)
    trainer.extend(extensions.ProgressBar(update_interval=3))

    # save model paramter
    trainer.extend(extensions.snapshot(), trigger=snapshot_trigger)
    trainer.extend(
        extensions.snapshot_object(
            model, 'model_iter_{.updater.iteration}.npz'),
        trigger=snapshot_trigger)
    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == "__main__":
    main()
