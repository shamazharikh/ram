import torch
import random
import os
import numpy as np
import argparse
from trainer import Trainer
from data_loader import get_MNIST_test_dataset, get_MNIST_train_val_dataset
from data_loader import get_test_loader, get_train_val_loader
from model import RecurrentAttention
from torch.optim.lr_scheduler import ReduceLROnPlateau
from callbacks import PlotCbk, ModelCheckpoint, LearningRateScheduler, EarlyStopping
import logging


def parse_args():
    def str2bool(v):
        return v.lower() in ('true', '1')

    import sys
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--num_glimpses', type=int, default=6, help='# of glimpses, i.e. BPTT iterations')
    parser.add_argument('--M', type=float, default=10, help='Monte Carlo sampling for valid and test sets')
    glimpse_arg = parser.add_argument_group('GlimpseNet Params')
    glimpse_arg.add_argument('--conv', type=str2bool, default=False, help='Use convolutional model')
    glimpse_arg.add_argument('--glimpse_hidden', type=int, default=128, help='hidden size of glimpse fc')
    glimpse_arg.add_argument('--loc_hidden', type=int, default=128, help='hidden size of loc fc')
    glimpse_arg.add_argument('--patch_size', type=int, default=8, help='size of extracted patch at highest res')
    glimpse_arg.add_argument('--num_patches', type=int, default=1, help='# of downscaled patches per glimpse')
    glimpse_arg.add_argument('--glimpse_scale', type=int, default=2, help='scale of successive patches')

    # LocationNet params
    location_arg = parser.add_argument_group('LocationNet Params')
    location_arg.add_argument('--std', type=float, default=0.17, help='gaussian policy standard deviation')

    # core_network params
    core_network_arg = parser.add_argument_group('core_network Params')
    core_network_arg.add_argument('--rnn_hidden', type=int, default=256, help='hidden size of the rnn')  # on purpose set equal to glimpse_hidden + loc_hidden, can be changed
    # data params
    data_arg = parser.add_argument_group('Data Params')
    data_arg.add_argument('--val_split', type=float, default=0.1, help='Proportion of training set used for validation')
    data_arg.add_argument('--num_workers', type=int, default=4, help='# of subprocesses to use for data loading')
    data_arg.add_argument('--random_split', type=str2bool, default=True, help='Whether to randomly split the train and valid indices')

    # training params
    train_arg = parser.add_argument_group('Training Params')
    train_arg.add_argument('--is_train', type=str2bool, default=True, help='Whether to train or test the model')
    train_arg.add_argument('--batch_size', type=int, default=32, help='# of images in each batch of data')
    train_arg.add_argument('--epochs', type=int, default=200, help='# of epochs to train for')
    train_arg.add_argument('--patience', type=int, default=100, help='Max # of epochs to wait for no validation improv')

    train_arg.add_argument('--momentum', type=float, default=0.5, help='Nesterov momentum value')
    train_arg.add_argument('--init_lr', type=float, default=0.001, help='Initial learning rate value')
    train_arg.add_argument('--min_lr', type=float, default=0.000001, help='Min learning rate value')
    train_arg.add_argument('--saturate_epoch', type=int, default=150, help='Epoch at which decayed lr will reach min_lr')
    # other params
    misc_arg = parser.add_argument_group('Misc.')
    misc_arg.add_argument('--use_gpu', type=str2bool, default=True, help="Whether to run on the GPU")
    misc_arg.add_argument('--device', type=int, default=1, help="GPU device to use")
    misc_arg.add_argument('--best', type=str2bool, default=True, help='Load best model or most recent for testing')
    misc_arg.add_argument('--random_seed', type=int, default=1, help='Seed to ensure reproducibility')
    misc_arg.add_argument('--data_dir', default='./data', help='Directory in which data is stored')
    misc_arg.add_argument('--ckpt_dir', default='./ckpt/conv_model', help='Directory in which to save model checkpoints')
    misc_arg.add_argument('--log_dir', default='./logs/', help='Directory in which Tensorboard logs wil be stored')
    misc_arg.add_argument('--use_tensorboard', type=str2bool, default=False, help='Whether to use tensorboard for visualization')
    misc_arg.add_argument('--resume', type=str2bool, default=False, help='Whether to resume training from checkpoint')
    misc_arg.add_argument('--print_freq', type=int, default=10, help='How frequently to print training details')
    misc_arg.add_argument('--plot_freq', type=int, default=1, help='How frequently to plot glimpses')
    misc_arg.add_argument('--plot_num_imgs', type=int, default=12, help='How many imgs to plot glimpses animiation')
    return parser.parse_args(sys.argv[1:])


def load_checkpoint(ckpt_dir, model, optimizer, best=False):
    print(1)
    if best:
        print ('model name',model.name)
        ckpt = torch.load(os.path.join(ckpt_dir, model.name+'_ckpt_best'))
    else:
        ckpt = torch.load(os.path.join(ckpt_dir, model.name))

    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(ckpt['optim_state_dict'])
    return ckpt['epoch']


if __name__ == '__main__':
    logger = logging.getLogger('RAM')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', "%m-%d %H:%M")
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    args = parse_args()
    # ensure reproducibility
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    kwargs = {}
    if args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.device)
        torch.cuda.manual_seed(args.random_seed)
        kwargs = {'num_workers': 1, 'pin_memory': True}

    if args.is_train:
        train_dataset, val_dataset = get_MNIST_train_val_dataset(args.data_dir)
        # train_dataset, val_dataset = get_MNIST_train_val_dataset('./data/MNIST')
        train_loader, val_loader = get_train_val_loader(
            train_dataset, val_dataset,
            val_split=args.val_split,
            random_split=args.random_split,
            batch_size=args.batch_size,
            **kwargs
        )
        args.num_class = train_loader.dataset.num_class
        args.num_channels = train_loader.dataset.num_channels

    else:
        test_dataset = get_MNIST_test_dataset(args.data_dir)
        test_loader = get_test_loader(test_dataset, args.batch_size, **kwargs)
        args.num_class = test_loader.dataset.num_class
        args.num_channels = test_loader.dataset.num_channels

    # build RAM model
    model = RecurrentAttention(args)
    if args.use_gpu:
        model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.init_lr, momentum=args.momentum)

    logger.info('Number of model parameters: {:,}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    trainer = Trainer(model, optimizer, watch=['acc'], val_watch=['acc'])

    if args.is_train:
        logger.info("Train on {} samples, validate on {} samples".format(len(train_loader.dataset), len(val_loader.dataset)))
        start_epoch = 0
        if args.resume:
            start_epoch = load_checkpoint(args.ckpt_dir, model, optimizer)

        trainer.train(train_loader, val_loader,
                      start_epoch=start_epoch,
                      epochs=args.epochs,
                      callbacks=[
                          PlotCbk(model, args.plot_num_imgs, args.plot_freq, args.use_gpu),
                          # TensorBoard(model, args.log_dir),
                          ModelCheckpoint(model, optimizer, args.ckpt_dir),
                          LearningRateScheduler(ReduceLROnPlateau(optimizer, 'min'), 'val_loss'),
                          EarlyStopping(model, patience=args.patience)
                      ])
    else:
        logger.info("Test on {} samples".format((len(test_loader))))
        print(args.ckpt_dir)
        epoch = load_checkpoint(args.ckpt_dir, model, False, best=True)
        print(epoch)
        trainer.test(test_loader, best=args.best)
