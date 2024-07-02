#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=100, help="rounds of training")
    parser.add_argument('--local_epochs', type=int, default=10, help="rounds of local training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1, help="the fraction of clients: C")
    parser.add_argument('--dropout', type=int, default=0, help="the percent of dropout clients")
    parser.add_argument('--bs', type=int, default=1024, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.999, help="learning rate decay each round")
    parser.add_argument('--random_seed', type=int, default=123, help="random seed of random generator")
    parser.add_argument('--upper_bound_of_random_number', type=int, 
                        default=123456789, help="upperbound of random number")

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")

    parser.add_argument('--dp_mechanism', type=str, default='Gaussian',
                        help='differential privacy mechanism')
    parser.add_argument('--k', type=float, default=0.5, help="The proportion of clients colluding with the server")
    parser.add_argument('--dp_epsilon', type=float, default=5,
                        help='differential privacy epsilon')
    parser.add_argument('--dp_delta', type=float, default=1e-5,
                        help='differential privacy delta')
    parser.add_argument('--dp_clip', type=float, default=10,
                        help='differential privacy clip')
    parser.add_argument('--dp_sample', type=float, default=1, help='sample rate for moment account')

    parser.add_argument('--serial', action='store_true', help='partial serial running to save the gpu memory')
    parser.add_argument('--serial_bs', type=int, default=128, help='partial serial running batch size')

    args = parser.parse_args()
    return args
