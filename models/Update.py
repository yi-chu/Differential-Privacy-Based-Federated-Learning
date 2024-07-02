#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from utils.dp_mechanism import cal_sensitivity, cal_sensitivity_MA, Laplace, Gaussian_Simple, Gaussian_MA
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdateDP(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.idxs_sample = np.random.choice(list(idxs), int(self.args.dp_sample * len(idxs)), replace=False)
        self.ldr_train = DataLoader(DatasetSplit(dataset, self.idxs_sample), batch_size=len(self.idxs_sample),
                                    shuffle=True)
        self.idxs = idxs
        self.times = self.args.epochs * self.args.frac
        self.lr = args.lr
        self.noise_scale = self.calculate_noise_scale()
        self.noise = 0
        # 客户端自己的公钥和私钥
        self.public_key = 0
        self.private_key = 0
        self.seed = {}
        self.rng = np.random.default_rng(np.random.randint(args.upper_bound_of_random_number))
        self.noise_generator = {}

    def get_public_key(self, args, p, g):
        self.private_key = self.rng.integers(args.upper_bound_of_random_number)
        self.public_key = pow(g, int(self.private_key), p)
        return self.public_key
    
    def generate_seeds(self, p, public_keys):
        # 清空上一轮生成的种子
        self.seed = {}
        for client_id, public_key in  public_keys.items():
            if client_id == self.id :
                continue
            self.seed[client_id] = pow(public_key, int(self.private_key), p)
            self.noise_generator[client_id] = np.random.default_rng(self.seed[client_id])

    def get_noise(self):
        return self.noise

    def set_id(self, id):
        self.id = id

    def set_variances(self, variances):
        self.variances = variances

    def calculate_noise_scale(self):
        if self.args.dp_mechanism == 'Laplace':
            epsilon_single_query = self.args.dp_epsilon / self.times
            return Laplace(epsilon=epsilon_single_query)
        elif self.args.dp_mechanism == 'Gaussian':
            epsilon_single_query = self.args.dp_epsilon / self.times
            delta_single_query = self.args.dp_delta / self.times
            return Gaussian_Simple(epsilon=epsilon_single_query, delta=delta_single_query)
        elif self.args.dp_mechanism == 'MA':
            return Gaussian_MA(epsilon=self.args.dp_epsilon, delta=self.args.dp_delta, q=self.args.dp_sample, epoch=self.times)
        elif self.args.dp_mechanism == 'DpSecureAggregation' or 'NISS':
            epsilon_single_query = self.args.dp_epsilon / self.times
            delta_single_query = self.args.dp_delta / self.times
            return Gaussian_Simple(epsilon=epsilon_single_query, delta=delta_single_query)

    def train(self, net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.args.lr_decay)
        loss_client = 0
        for _ in range(self.args.local_epochs):
            for images, labels in self.ldr_train:
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                if self.args.dp_mechanism != 'no_dp':
                    self.clip_gradients(net)
                optimizer.step()
                scheduler.step()
                # add noises to parameters
                if self.args.dp_mechanism != 'no_dp':
                    self.add_noise(net)
                loss_client = loss.item()
            self.lr = scheduler.get_last_lr()[0]
        return net.state_dict(), loss_client

    def clip_gradients(self, net):
        if self.args.dp_mechanism == 'Laplace':
            # Laplace use 1 norm
            self.per_sample_clip(net, self.args.dp_clip, norm=1)
        elif self.args.dp_mechanism == 'Gaussian' or self.args.dp_mechanism == 'MA':
            # Gaussian use 2 norm
            self.per_sample_clip(net, self.args.dp_clip, norm=2)

    def per_sample_clip(self, net, clipping, norm):
        grad_samples = [x.grad_sample for x in net.parameters()]
        per_param_norms = [
            g.reshape(len(g), -1).norm(norm, dim=-1) for g in grad_samples
        ]
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(norm, dim=1)
        per_sample_clip_factor = (
            torch.div(clipping, (per_sample_norms + 1e-6))
        ).clamp(max=1.0)
        for grad in grad_samples:
            factor = per_sample_clip_factor.reshape(per_sample_clip_factor.shape + (1,) * (grad.dim() - 1))
            grad.detach().mul_(factor.to(grad.device))
        # average per sample gradient after clipping and set back gradient
        for param in net.parameters():
            param.grad = param.grad_sample.detach().mean(dim=0)

    def get_variance(self):
        sensitivity = cal_sensitivity(self.lr, self.args.dp_clip, len(self.idxs_sample))
        return sensitivity * self.calculate_noise_scale()

    def add_noise(self, net):
        sensitivity = cal_sensitivity(self.lr, self.args.dp_clip, len(self.idxs_sample))
        state_dict = net.state_dict()
        if self.args.dp_mechanism == 'Laplace':
            for k, v in state_dict.items():
                state_dict[k] += torch.from_numpy(np.random.laplace(loc=0, scale=sensitivity * self.noise_scale,
                                                                    size=v.shape)).to(self.args.device)
        elif self.args.dp_mechanism == 'Gaussian':
            for k, v in state_dict.items():
                noise = np.random.normal(loc=0, scale=sensitivity * self.noise_scale,
                                                                   size=v.shape)
                self.noise += noise.sum()
                state_dict[k] += torch.from_numpy(noise).to(self.args.device)
        elif self.args.dp_mechanism == 'MA':
            sensitivity = cal_sensitivity_MA(self.args.lr, self.args.dp_clip, len(self.idxs_sample))
            for k, v in state_dict.items():
                state_dict[k] += torch.from_numpy(np.random.normal(loc=0, scale=sensitivity * self.noise_scale,
                                                                   size=v.shape)).to(self.args.device)
        elif self.args.dp_mechanism == 'DpSecureAggregation':
            for k, v in state_dict.items():
                for client_id, variance in self.variances.items():
                    if client_id == self.id:
                        continue
                    if self.id > client_id: 
                        noise = self.noise_generator[client_id].normal(loc=0, 
                                                scale=(1/(1-self.args.k))*self.variances[self.id]/(len(self.variances)-1),
                                                size=v.shape)
                        self.noise += noise.sum()
                        state_dict[k] += torch.from_numpy(noise).to(self.args.device)
                        noise = self.noise_generator[client_id].normal(loc=0, 
                                                scale=(1/(1-self.args.k))*variance/(len(self.variances)-1),
                                                size=v.shape)
                        self.noise -= noise.sum()
                        state_dict[k] -= torch.from_numpy(noise).to(self.args.device)
                    else:
                        noise = self.noise_generator[client_id].normal(loc=0, 
                                                scale=(1/(1-self.args.k))*variance/(len(self.variances)-1),
                                                size=v.shape)
                        self.noise -= noise.sum()
                        state_dict[k] -= torch.from_numpy(noise).to(self.args.device)
                        noise = self.noise_generator[client_id].normal(loc=0, 
                                                scale=(1/(1-self.args.k))*self.variances[self.id]/(len(self.variances)-1),
                                                size=v.shape)
                        self.noise += noise.sum()
                        state_dict[k] += torch.from_numpy(noise).to(self.args.device)
        elif self.args.dp_mechanism == "NISS":
            noise_factor = np.random.normal(loc=1,scale=max(0,self.args.k*2-1))
            for k, v in state_dict.items():
                for client_id, variance in self.variances.items():
                    if client_id == self.id:
                        continue
                    if self.id > client_id: 
                        # 自己的噪音
                        noise = self.noise_generator[client_id].normal(loc=0, 
                                                scale=self.variances[self.id]/(len(self.variances)-1),
                                                size=v.shape)
                        self.noise += noise.sum()
                        state_dict[k] += torch.from_numpy(noise).to(self.args.device)
                        # 对方的噪音
                        noise = self.noise_generator[client_id].normal(loc=0, 
                                                scale=variance/(len(self.variances)-1),
                                                size=v.shape) * noise_factor
                        self.noise -= noise.sum()
                        state_dict[k] -= torch.from_numpy(noise).to(self.args.device)
                    else:
                        # 对方的噪音
                        noise = self.noise_generator[client_id].normal(loc=0, 
                                                scale=variance/(len(self.variances)-1),
                                                size=v.shape) * noise_factor
                        self.noise -= noise.sum()
                        state_dict[k] -= torch.from_numpy(noise).to(self.args.device)
                        # 自己的噪音
                        noise = self.noise_generator[client_id].normal(loc=0, 
                                                scale=self.variances[self.id]/(len(self.variances)-1),
                                                size=v.shape)
                        self.noise += noise.sum()
                        state_dict[k] += torch.from_numpy(noise).to(self.args.device)
        net.load_state_dict(state_dict)


class LocalUpdateDPSerial(LocalUpdateDP):
    def __init__(self, args, dataset=None, idxs=None):
        super().__init__(args, dataset, idxs)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.args.lr_decay)
        losses = 0
        for _ in range(self.args.local_epochs):
            for images, labels in self.ldr_train:
                net.zero_grad()
                index = int(len(images) / self.args.serial_bs)
                total_grads = [torch.zeros(size=param.shape).to(self.args.device) for param in net.parameters()]
                for i in range(0, index + 1):
                    net.zero_grad()
                    start = i * self.args.serial_bs
                    end = (i+1) * self.args.serial_bs if (i+1) * self.args.serial_bs < len(images) else len(images)
                    # print(end - start)
                    if start == end:
                        break
                    image_serial_batch, labels_serial_batch \
                        = images[start:end].to(self.args.device), labels[start:end].to(self.args.device)
                    log_probs = net(image_serial_batch)
                    loss = self.loss_func(log_probs, labels_serial_batch)
                    loss.backward()
                    if self.args.dp_mechanism != 'no_dp':
                        self.clip_gradients(net)
                    grads = [param.grad.detach().clone() for param in net.parameters()]
                    for idx, grad in enumerate(grads):
                        total_grads[idx] += torch.mul(torch.div((end - start), len(images)), grad)
                    losses += loss.item() * (end - start)
                for i, param in enumerate(net.parameters()):
                    param.grad = total_grads[i]
                optimizer.step()
                scheduler.step()
                # add noises to parameters
                self.lr = scheduler.get_last_lr()[0]
        if self.args.dp_mechanism != 'no_dp':
            self.add_noise(net)
        
        return net.state_dict(), losses / len(self.idxs_sample)
