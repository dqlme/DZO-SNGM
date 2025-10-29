
"""
分布式零阶优化算法集合 - CUDA版本
包含:  Fed-ZO-SGD, DZO-SNGM,  Fed-ZO-Adam,  Fed-ZO-SVRG,   Fed-ZO-signSGD, FedZO
支持GPU加速计算
"""

import numpy as np
import torch
from copy import deepcopy
import time
from typing import Dict, List, Optional, Tuple

def move_to_device(data, device):
    """将数据移动到指定设备"""
    if isinstance(data, np.ndarray):
        return torch.FloatTensor(data).to(device)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data


def run_dzo_sngm_cuda(X_train, Y_train, X_test, Y_test, Ntrain, n, nWorkers, 
                  batchsize, data_idxs, nLocalSteps, global_epoch, sigma0, theta, model,
                  beta1=0.9, beta2=0.999, epsilon=1e-8, momentum_factor=0.5, adaptive_sigma=True,
                  device='cuda'):
    """
    DZO-SNGM: CUDA版本
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Running DZO-SNGM (our proposed method) on {device}...")
    
    # 将数据移动到设备
    X_train = move_to_device(X_train, device)
    Y_train = move_to_device(Y_train, device)
    X_test = move_to_device(X_test, device)
    Y_test = move_to_device(Y_test, device)
    
    loss_, accuracy_ = [], []
    
    # 为每个工作节点准备数据
    X_trains, Y_trains, Nis = [], [], []    
    for i in range(nWorkers):
        Nis.append(len(data_idxs[i]))
        X_trains.append(X_train[data_idxs[i], :])     
        Y_trains.append(Y_train[data_idxs[i]])       
    
    # 初始化动量和二阶矩
    # m = np.zeros(n)
    # v = np.zeros(n)
    
    for t in range(global_epoch):
        if t == 0:
            train_loss = model.loss(X_train, Y_train, theta)
            predict = model.predict(X_test, theta)
            test_acc = model.score(predict, Y_test)                
            loss_.append(train_loss)
            accuracy_.append(test_acc)
            print(f"round {0}, training loss = {round(train_loss, 6)}, test accuracy = {round(test_acc, 6)}")
        
        worker_solutions = np.zeros((nWorkers, n))
       
        sigma0 = 1e-3
        sigma = sigma0
        for i in range(nWorkers):
            local_solution = deepcopy(theta)
            sizes = local_solution.shape
            for l in range(nLocalSteps):
                # 生成随机扰动方向
                v = np.random.normal(size=sizes)
                v_rand = v * sigma  # 直接使用sigma调整扰动规模
                
                # 随机选择小批次
                minibatchidx = np.random.choice(Nis[i], min(batchsize, Nis[i]), replace=False)
                local_fobj = lambda w: model.evaluate_func(X_trains[i][minibatchidx, :], Y_trains[i][minibatchidx], w)
                
                # 对称差分梯度估计
                f_pos = local_fobj(local_solution + v_rand)
                f_neg = local_fobj(local_solution - v_rand)
                g_trial = v * (f_pos - f_neg) / (2 * sigma)
                
                local_solution = local_solution -  sigma0 * g_trial
            worker_solutions[i, :] = local_solution
        d = theta - np.mean(worker_solutions, axis=0)
        if t == 0:
            m = d
        else:
            a = 1 / np.sqrt(t+1)
            # a=0.9
            m = (1-a) * m + a * d
        theta = theta -  0.1 * (m / (np.linalg.norm(m) + 1e-8))            
        
        
        # 计算训练损失
        train_loss = model.loss(X_train, Y_train, theta)
        
        # 计算测试准确率
        predict = model.predict(X_test, theta)
        test_acc = model.score(predict, Y_test)
        
        loss_.append(train_loss)
        accuracy_.append(test_acc)
        
        print(f"round {t+1}, training loss = {round(train_loss, 6)}, test accuracy = {round(test_acc, 6)}")
    
    return loss_, accuracy_
