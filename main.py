#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import random
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import os
import time

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid,cifar_noniid
from utils.options import args_parser
from models.Update import LocalUpdateDP, LocalUpdateDPSerial
from models.Nets import MLP, CNNMnist, CNNCifar, CNNFemnist, CharLSTM, VGG11
from models.Fed import FedAvg, FedWeightAvg
from models.test import test_img
from utils.dataset import FEMNIST, ShakeSpeare
from opacus.grad_sample import GradSampleModule
from utils.diffie_hellman import get_params
from threading import Thread
from utils import *
from user import User
from server import *

if __name__ == '__main__':
    # parse args

    args = args_parser()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    dict_users = {}
    dataset_train, dataset_test = None, None

    rootpath = './log'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)
    datfile = open(rootpath + '/data_size_file_fed_{}_{}_{}_iid{}_dp_{}_epsilon_{}_k_{}_drop_{}_{}.dat'.
                   format(args.dataset, args.model, args.epochs, args.iid,
                          args.dp_mechanism, args.dp_epsilon, args.k, args.dropout, args.num_users), "w")

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        args.num_channels = 1
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        #trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        args.num_channels = 3
        trans_cifar_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trans_cifar_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar_train)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar_test)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_noniid(dataset_train, args.num_users)
    elif args.dataset == 'fashion-mnist':
        args.num_channels = 1
        trans_fashion_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset_train = datasets.FashionMNIST('./data/fashion-mnist', train=True, download=True,
                                              transform=trans_fashion_mnist)
        dataset_test  = datasets.FashionMNIST('./data/fashion-mnist', train=False, download=True,
                                              transform=trans_fashion_mnist)
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'femnist':
        args.num_channels = 1
        dataset_train = FEMNIST(train=True)
        dataset_test = FEMNIST(train=False)
        dict_users = dataset_train.get_client_dic()
        args.num_users = len(dict_users)
        if args.iid:
            exit('Error: femnist dataset is naturally non-iid')
        else:
            print("Warning: The femnist dataset is naturally non-iid, you do not need to specify iid or non-iid")
    elif args.dataset == 'shakespeare':
        dataset_train = ShakeSpeare(train=True)
        dataset_test = ShakeSpeare(train=False)
        dict_users = dataset_train.get_client_dic()
        args.num_users = len(dict_users)
        if args.iid:
            exit('Error: ShakeSpeare dataset is naturally non-iid')
        else:
            print("Warning: The ShakeSpeare dataset is naturally non-iid, you do not need to specify iid or non-iid")
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    net_glob = None
    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and (args.dataset == 'mnist' or args.dataset == 'fashion-mnist'):
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.dataset == 'femnist' and args.model == 'cnn':
        net_glob = CNNFemnist(args=args).to(args.device)
    elif args.dataset == 'shakespeare' and args.model == 'lstm':
        net_glob = CharLSTM().to(args.device)
    elif args.model == 'VGG11':
        net_glob = VGG11().to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')

    # use opacus to wrap model to clip per sample gradient
    if args.dp_mechanism != 'no_dp':
        net_glob = GradSampleModule(net_glob)
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()
    all_clients = list(range(args.num_users))

    # training
    acc_test = []
    if args.serial:
        clients = [LocalUpdateDPSerial(args=args, dataset=dataset_train, idxs=dict_users[i]) for i in range(args.num_users)]
    else:
        clients = [LocalUpdateDP(args=args, dataset=dataset_train, idxs=dict_users[i]) for i in range(args.num_users)]
    m, loop_index = max(int(args.frac * args.num_users), 1), int(1 / args.frac)
    upload = 0
    download = 0
    add_variances = 0
    for iter in range(args.epochs):
        t_start = time.time()
        w_locals, loss_locals, weight_locols = [], [], []
        # round-robin selection
        begin_index = (iter % loop_index) * m
        end_index = begin_index + m
        idxs_users = all_clients[begin_index:end_index]
        variances = {}
        if args.dp_mechanism == "DpSecureAggregation" or "AGG":
            # 用时间作为随机数种子
            np.random.seed(int(time.time()))
            # 为所有客户端设置id
            for idx in idxs_users:
                clients[idx].set_id(idx)

            # 获取dh算法参数
            p, g = get_params(args.random_seed)
            public_keys = {}
            # 客户端生成自己的私钥和公钥，并将公钥发给服务端
            for idx in idxs_users:
                public_keys[idx] = clients[idx].get_public_key(args, p, g)
            upload += len(pickle.dumps(public_keys))
            download += len(pickle.dumps(public_keys))*len(idxs_users)

            # 客户端通过服务端发来的公钥列表，生成对于的随机数种子列表
            for idx in idxs_users:
                clients[idx].generate_seeds(p, public_keys)

            # 获取每个客户端所生成的噪音的方差
            for idx in idxs_users:
                variances[idx] = clients[idx].get_variance();
            for idx in idxs_users:
                clients[idx].set_variances(variances)
            if add_variances == 0 and args.dp_mechanism == "DpSecureAggregation":
                add_variances = 1
                upload += len(pickle.dumps(variances))
                download += len(pickle.dumps(variances))*len(idxs_users)

        # 随机选择一部分掉线客户端
        if args.dp_mechanism == "DpSecureAggregation" or "NISS":
            dropout = np.random.choice(idxs_users, int(len(idxs_users)*(args.dropout/100)))

        for idx in idxs_users:
            if idx in dropout:
                continue
            local = clients[idx]
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            weight_locols.append(len(dict_users[idx]))

        # update global weights
        w_glob = FedWeightAvg(w_locals, weight_locols)
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        download += len(pickle.dumps(w_glob))*len(idxs_users)
        upload += len(pickle.dumps(w_glob))*len(idxs_users)
        if args.dp_mechanism == "NISS":
            download += len(pickle.dumps(w_glob))*len(idxs_users)*len(idxs_users)
            upload += len(pickle.dumps(w_glob))*len(idxs_users)*len(idxs_users)

        # print accuracy
        net_glob.eval()
        acc_t, loss_t = test_img(net_glob, dataset_test, args)
        t_end = time.time()
        print("Round {:3d},Testing accuracy: {:.2f},Time:  {:.2f}s".format(iter, acc_t, t_end - t_start))

        acc_test.append(acc_t.item())
        print("upload = ",upload, "download = ", download)
        datfile.write("epoch= {}, acc = {}\n".format(iter, acc_t.item()))
        datfile.write("upload= {}, download = {}\n".format(upload, download))

    noise = 0
    if args.dp_mechanism == "DpSecureAggregation" or "NISS":
        for idx in idxs_users:
            noise += clients[idx].get_noise()
            print(clients[idx].get_noise())
        print(noise)

    if not os.path.exists(rootpath):
        os.makedirs(rootpath)
    accfile = open(rootpath + '/accfile_fed_{}_{}_{}_iid{}_dp_{}_epsilon_{}_k_{}_drop_{}.dat'.
                   format(args.dataset, args.model, args.epochs, args.iid,
                          args.dp_mechanism, args.dp_epsilon, args.k, args.dropout), "w")

    for ac in acc_test:
        sac = str(ac)
        accfile.write(sac)
        accfile.write('\n')
    accfile.close()

    # plot loss curve
    plt.figure()
    plt.plot(range(len(acc_test)), acc_test)
    plt.ylabel('test accuracy')
    plt.savefig(rootpath + '/fed_{}_{}_{}_C{}_iid{}_dp_{}_epsilon_{}_k_{}_drop_{}_acc.png'.format(
        args.dataset, args.model, args.epochs, args.frac, args.iid, args.dp_mechanism, args.dp_epsilon, args.k, args.dropout))



