
"""
分布式零阶优化算法集合 - CUDA版本
包含: DZO-SGD, DZO-SNGM, DZO-Adam, DZO-SVRG, DES, DZO-signSGD, Fed-ZO-SGD
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

def run_dzo_sngm_cuda_new(X_train, Y_train, X_test, Y_test, Ntrain, n, nWorkers, 
                  batchsize, data_idxs, nLocalSteps, global_epoch, sigma0, theta, model,
                  beta1=0.9, beta2=0.999, epsilon=1e-8, momentum_factor=0.5, adaptive_sigma=True,
                  device='cuda'):
    """
    DZO-SNGM: 分布式零阶随机自然梯度动量法 - CUDA版本
    结合了自然梯度、动量和自适应学习率
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
    m = np.zeros(n)
    v = np.zeros(n)
    
    for t in range(global_epoch):
        if t == 0:
            train_loss = model.loss(X_train, Y_train, theta)
            predict = model.predict(X_test, theta)
            test_acc = model.score(predict, Y_test)                
            loss_.append(train_loss)
            accuracy_.append(test_acc)
            print(f"round {0}, training loss = {round(train_loss, 6)}, test accuracy = {round(test_acc, 6)}")
        
        worker_solutions = np.zeros((nWorkers, n))
        
        # 自适应参数
        if adaptive_sigma:
            sigma = sigma0 / (t + 1)**0.5  # 自适应步长
            mu = min(1e-4, sigma0 / (t + 1)**0.75)  # 自适应平滑参数
        else:
            sigma = sigma0
            mu = 1e-4
        
        for i in range(nWorkers):
            local_solution = deepcopy(theta)
            local_m = deepcopy(m)
            local_v = deepcopy(v)
            
            for l in range(nLocalSteps):
                # 生成随机扰动方向
                v_rand = np.random.normal(size=n)
                v_rand = v_rand / (np.linalg.norm(v_rand) + 1e-8)
                
                # 随机选择小批次
                minibatchidx = np.random.choice(Nis[i], min(batchsize, Nis[i]), replace=False)
                local_fobj = lambda w: model.evaluate_func(X_trains[i][minibatchidx, :], Y_trains[i][minibatchidx], w)
                
                # 对称差分梯度估计
                f_pos = local_fobj(local_solution + v_rand * mu)
                f_neg = local_fobj(local_solution - v_rand * mu)
                g_trial = v_rand * (f_pos - f_neg) / (2 * mu)
                
                # SNGM更新规则 - 结合自然梯度和动量
                # 一阶矩估计 (动量)
                local_m = beta1 * local_m + (1 - beta1) * g_trial
                
                # 二阶矩估计 (自适应学习率)
                local_v = beta2 * local_v + (1 - beta2) * (g_trial ** 2)
                
                # 偏差修正
                step_count = t * nLocalSteps + l + 1
                m_hat = local_m / (1 - beta1 ** step_count)
                v_hat = local_v / (1 - beta2 ** step_count)
                
                # 自然梯度方向 - 使用Fisher信息矩阵的近似
                natural_grad = m_hat / (np.sqrt(v_hat) + epsilon)
                
                # 自适应步长调整
                grad_norm = np.linalg.norm(natural_grad)
                if grad_norm > 0:
                    adaptive_step = sigma / (1 + grad_norm)
                    local_solution = local_solution - adaptive_step * natural_grad
            
            worker_solutions[i, :] = local_solution
        
        # 全局更新 - 带动量的聚合
        theta = np.mean(worker_solutions, axis=0)
        
        # 计算训练损失
        train_loss = model.loss(X_train, Y_train, theta)
        
        # 计算测试准确率
        predict = model.predict(X_test, theta)
        test_acc = model.score(predict, Y_test)
        
        loss_.append(train_loss)
        accuracy_.append(test_acc)
        
        print(f"round {t+1}, training loss = {round(train_loss, 6)}, test accuracy = {round(test_acc, 6)}")
    
    return loss_, accuracy_



def run_dzo_sngm_cuda(X_train, Y_train, X_test, Y_test, Ntrain, n, nWorkers, 
                  batchsize, data_idxs, nLocalSteps, global_epoch, sigma0, theta, model,
                  beta1=0.9, beta2=0.999, epsilon=1e-8, momentum_factor=0.5, adaptive_sigma=True,
                  device='cuda'):
    """
    DZO-SNGM: 分布式零阶随机自然梯度动量法 - CUDA版本
    结合了自然梯度、动量和自适应学习率
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
        # 自适应参数
        # if adaptive_sigma:
            # sigma = sigma0 / (t + 1)**0.75  # 自适应步长
            # mu = min(1e-4, sigma0 / (t + 1)**0.75)  # 自适应平滑参数
        # else:
        #     sigma = sigma0
        #     mu = 1e-4
        # sigma = 1e-3
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




def run_dzo_sgd_cuda(X_train, Y_train, X_test, Y_test, Ntrain, n, nWorkers, 
                batchsize, data_idxs, nLocalSteps, global_epoch, sigma0, theta, model,
                momentum=0.9, gradient_clip=1.0, adaptive_sigma=True, device='cuda'):
    """
    分布式ZO-SGD算法 - CUDA版本
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Running DZO-SGD on {device}...")
    
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
    
    # 初始化动量
    # velocity = np.zeros(n)
    sizes = theta.shape
    for t in range(global_epoch):
        if t == 0:
            train_loss = model.loss(X_train, Y_train, theta)
            predict = model.predict(X_test, theta)
            test_acc = model.score(predict, Y_test)                
            loss_.append(train_loss)
            accuracy_.append(test_acc)
            print(f"round {0}, training loss = {round(train_loss, 6)}, test accuracy = {round(test_acc, 6)}")
        
        worker_solutions = np.zeros((nWorkers, n))
        # sigma0 = 0.0001
        # sigma = 0.00001
        sigma0 = 1e-3
        sigma = 1e-3
        # 自适应参数
        # if adaptive_sigma:
        #     sigma = sigma0 / (t + 1)**0.3
        #     mu = min(1e-4, sigma0 / (t + 1)**0.5)
        # else:
        #     sigma = sigma0
        #     mu = 1e-4
        
        for i in range(nWorkers):
            local_solution = deepcopy(theta)
            
            for l in range(nLocalSteps):
                # 生成随机扰动方向
                v = np.random.normal(size=sizes)
                # v = v / (np.linalg.norm(v) + 1e-8)
                v_rand = v * sigma
                
                # 随机选择小批次
                minibatchidx = np.random.choice(Nis[i], min(batchsize, Nis[i]), replace=False)
                local_fobj = lambda w: model.evaluate_func(X_trains[i][minibatchidx, :], Y_trains[i][minibatchidx], w)
                
                # 对称差分梯度估计
                f_plus = local_fobj(local_solution + v_rand)
                f_minus = local_fobj(local_solution - v_rand)
                g_trial = v * (f_plus - f_minus) / (2 * sigma)
                
                # 参数更新
                local_solution = local_solution - sigma0 * g_trial
            
            worker_solutions[i, :] = local_solution
        
        # 全局更新
        theta = np.mean(worker_solutions, axis=0)
        
        # 计算性能指标
        train_loss = model.loss(X_train, Y_train, theta)
        predict = model.predict(X_test, theta)
        test_acc = model.score(predict, Y_test)
        
        loss_.append(train_loss)
        accuracy_.append(test_acc)
        
        print(f"round {t+1}, training loss = {round(train_loss, 6)}, test accuracy = {round(test_acc, 6)}")
    
    return loss_, accuracy_

def run_dzo_adam_cuda(X_train, Y_train, X_test, Y_test, Ntrain, n, nWorkers, 
                 batchsize, data_idxs, nLocalSteps, global_epoch, sigma0, theta, model,
                 beta1=0.9, beta2=0.999, epsilon=1e-8, alpha=0.001, device='cuda'):
    """
    分布式ZO-Adam算法 - CUDA版本
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Running DZO-Adam on {device}...")
    
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
    
    # Adam状态变量
    m = np.zeros(n)
    v = np.zeros(n)
    
    for t in range(global_epoch):
        if t == 0:
            train_loss = model.loss(X_train, Y_train, theta)
            predict = model.predict(X_test, theta)
            test_acc = model.score(predict, Y_test)                
            loss_.append(train_loss)
            accuracy_.append(test_acc)
            print(f"round {0}, training loss = {round(train_loss, 6)}, test accuracy = {round(test_acc, 6)}")
        
        worker_solutions = np.zeros((nWorkers, n))
        mu = 1e-3  # 高斯平滑参数
        sigma = 1e-3
        # alpha=0.001
        # python main.py --dataset mnist --method DZO-Adam --use_cuda
        for i in range(nWorkers):
            local_solution = deepcopy(theta)
            local_m = deepcopy(m)
            local_v = deepcopy(v)
            
            for l in range(nLocalSteps):
                # 自适应步长
                # sigma = sigma0 / (t + 1)**0.5 / (l + 1)**0.5
                
                # 生成随机扰动方向
                v_rand = np.random.normal(size=n)
                
                # 随机选择小批次
                minibatchidx = np.random.choice(Nis[i], min(batchsize, Nis[i]), replace=False)
                local_fobj = lambda w: model.evaluate_func(X_trains[i][minibatchidx, :], Y_trains[i][minibatchidx], w)
                
                # 零阶梯度估计
                pre_local_fobj = local_fobj(local_solution - v_rand * mu)
                current_local_fobj = local_fobj(local_solution + v_rand * mu)
                g_trial = v_rand * (current_local_fobj - pre_local_fobj) / (2 * mu)
                
                # Adam更新规则
                local_m = beta1 * local_m + (1 - beta1) * g_trial
                local_v = beta2 * local_v + (1 - beta2) * (g_trial ** 2)
                
                # 偏差修正
                step_count = t * nLocalSteps + l + 1
                m_hat = local_m / (1 - beta1 ** step_count)
                v_hat = local_v / (1 - beta2 ** step_count)
                
                # 参数更新
                effective_lr = sigma 
                local_solution = local_solution - effective_lr * m_hat / (np.sqrt(v_hat) + epsilon)
            
            worker_solutions[i, :] = local_solution
        
        # 全局更新
        theta = np.mean(worker_solutions, axis=0)
        
        # 计算性能指标
        train_loss = model.loss(X_train, Y_train, theta)
        predict = model.predict(X_test, theta)
        test_acc = model.score(predict, Y_test)
        
        loss_.append(train_loss)
        accuracy_.append(test_acc)
        
        print(f"round {t+1}, training loss = {round(train_loss, 6)}, test accuracy = {round(test_acc, 6)}")
    
    return loss_, accuracy_

def run_dzo_svrg_cuda(X_train, Y_train, X_test, Y_test, Ntrain, n, nWorkers, 
                  batchsize, data_idxs, nLocalSteps, global_epoch, sigma0, theta, model,
                  inner_loop_size=5, momentum=0.9, device='cuda'):
    """
    分布式ZO-SVRG算法 - CUDA版本
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Running DZO-SVRG on {device}...")
    
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
    
    # 初始化
    w_t = deepcopy(theta)
    sizes = w_t.shape
    for t in range(global_epoch):
        if t == 0:
            train_loss = model.loss(X_train, Y_train, w_t)
            predict = model.predict(X_test, w_t)
            test_acc = model.score(predict, Y_test)                
            loss_.append(train_loss)
            accuracy_.append(test_acc)
            print(f"round {0}, training loss = {round(train_loss, 6)}, test accuracy = {round(test_acc, 6)}")
        
        # 计算全梯度估计（使用小批次近似）
        full_grad_est = np.zeros(n)
        # mu_full = 1e-2
        # mu_full = 1e-4
        mu_full = 1e-3
        for i in range(nWorkers):
            # 为每个工作节点选择代表性样本
            sample_idx = np.random.choice(Nis[i], min(batchsize, Nis[i]), replace=False)
            local_fobj = lambda w: model.evaluate_func(X_trains[i][sample_idx, :], Y_trains[i][sample_idx], w)
            
            # 梯度估计
            v = np.random.normal(size=sizes)
            # v = v / (np.linalg.norm(v) + 1e-8)
            v_rand = v * mu_full
            # v_rand = v_rand / (np.linalg.norm(v_rand) + 1e-8)
            
            f_plus = local_fobj(w_t + v_rand)
            f_minus = local_fobj(w_t - v_rand)
            grad_est = v * (f_plus - f_minus) / (2 * mu_full)
            
            full_grad_est += grad_est / nWorkers
        
        # 内循环 - SVRG更新
        w_t_new = deepcopy(w_t)
        
        for inner_iter in range(inner_loop_size):
            worker_solutions = np.zeros((nWorkers, n))
            # sigma_inner = sigma0 / (t + 1)**0.5 / (inner_iter + 1)**0.2
            # mu = 1e-2 for mnist
            # mu = 1e-4
            mu = 1e-3   
            for i in range(nWorkers):
                local_w = deepcopy(w_t_new)
                
                # 随机选择小批次
                minibatchidx = np.random.choice(Nis[i], min(batchsize, Nis[i]), replace=False)
                local_fobj = lambda w: model.evaluate_func(X_trains[i][minibatchidx, :], Y_trains[i][minibatchidx], w)
                
                # 随机梯度估计
                v = np.random.normal(size=sizes)
                # v = v / (np.linalg.norm(v) + 1e-8)
                v_rand = v * mu
        
                
                
                f_plus = local_fobj(local_w + v_rand)
                f_minus = local_fobj(local_w - v_rand)
                g_rand = v * (f_plus - f_minus) / (2 * mu)
                
                # 计算w_t处的梯度估计
                f_plus_old = local_fobj(w_t + v_rand)
                f_minus_old = local_fobj(w_t - v_rand)
                g_old = v * (f_plus_old - f_minus_old) / (2 * mu)
                
                # SVRG梯度
                svrg_grad = g_rand - g_old + full_grad_est
                
                # 动量更新
                svrg_grad = momentum * svrg_grad + (1 - momentum) * svrg_grad
                
                # 参数更新
                local_w = local_w - 0.001* svrg_grad
                
                worker_solutions[i, :] = local_w
            
            # 更新w_t_new
            w_t_new = np.mean(worker_solutions, axis=0)
        
        # 更新主参数
        w_t = deepcopy(w_t_new)
        
        # 计算性能指标
        train_loss = model.loss(X_train, Y_train, w_t)
        predict = model.predict(X_test, w_t)
        test_acc = model.score(predict, Y_test)
        
        loss_.append(train_loss)
        accuracy_.append(test_acc)
        
        print(f"round {t+1}, training loss = {round(train_loss, 6)}, test accuracy = {round(test_acc, 6)}")
    
    return loss_, accuracy_

def run_des_cuda(X_train, Y_train, X_test, Y_test, Ntrain, n, nWorkers,
            batchsize, data_idxs, nLocalSteps, global_epoch, sigma0, theta, model,
            momentum_weight=0.5, device='cuda'):
    """
    DES: Distributed Evolution Strategy算法 - CUDA版本
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Running DES on {device}...")
    
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
    
    # 动量向量
    m = np.zeros(n)
    sizes = theta.shape
    for t in range(global_epoch):
        if t == 0:
            train_loss = model.loss(X_train, Y_train, theta)
            predict = model.predict(X_test, theta)
            test_acc = model.score(predict, Y_test)
            loss_.append(train_loss)
            accuracy_.append(test_acc)
            print(f"round {0}, training loss = {round(train_loss, 6)}, test accuracy = {round(test_acc, 6)}")
        
        # 自适应sigma
        sigma0 = 1
        sigma_t = sigma0 / (t + 1)**0.25
        # sigma = 0.001
        worker_solutions = np.zeros((nWorkers, n))
        
        for i in range(nWorkers):
            local_solution = deepcopy(theta)
            f_best = model.evaluate_func(X_trains[i][:min(batchsize, Nis[i])],
                                       Y_trains[i][:min(batchsize, Nis[i])], local_solution)
            
            # 局部进化策略
            for l in range(nLocalSteps):
                sigma = sigma_t / (l + 1)**0.5
                # sigma = 0.0001
                
                # 生成随机试验点
                trial_perturbation = np.random.normal(size=sizes) * sigma
                local_trial = local_solution + trial_perturbation
                
                # 评估试验点
                f_trial = model.evaluate_func(X_trains[i][:min(batchsize, Nis[i])],
                                            Y_trains[i][:min(batchsize, Nis[i])], local_trial)
                
                # 选择更好的解
                if f_trial < f_best:
                    f_best = f_trial
                    local_solution = deepcopy(local_trial)
            
            worker_solutions[i, :] = local_solution
        momentum_weight = 0.1
        # 动量更新
        d =  np.mean(worker_solutions, axis=0) - theta
        if t == 0:
            m = d
        else:
            m = momentum_weight * m + (1 - momentum_weight) * d
        
        theta = theta +  0.1 * m
        
        # 计算性能指标
        train_loss = model.loss(X_train, Y_train, theta)
        predict = model.predict(X_test, theta)
        test_acc = model.score(predict, Y_test)
        
        loss_.append(train_loss)
        accuracy_.append(test_acc)
        
        print(f"round {t+1}, training loss = {round(train_loss, 6)}, test accuracy = {round(test_acc, 6)}")
    
    return loss_, accuracy_

def run_dzo_signsgd_cuda(X_train, Y_train, X_test, Y_test, Ntrain, n, nWorkers,
                    batchsize, data_idxs, nLocalSteps, global_epoch, sigma0, theta, model,
                    sign_momentum=0.9, device='cuda'):
    """
    DZO-signSGD算法 - CUDA版本
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Running DZO-signSGD on {device}...")
    
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
    sizes = theta.shape
    for t in range(global_epoch):
        if t == 0:
            train_loss = model.loss(X_train, Y_train, theta)
            predict = model.predict(X_test, theta)
            test_acc = model.score(predict, Y_test)
            loss_.append(train_loss)
            accuracy_.append(test_acc)
            print(f"round {0}, training loss = {round(train_loss, 6)}, test accuracy = {round(test_acc, 6)}")
        
        worker_solutions = np.zeros((nWorkers, n))
        mu = 1e-3  # 高斯平滑参数
        sigma = 1e-3
        # sigma = 0.0001
        # mu = 0.00001        
        for i in range(nWorkers):
            local_solution = deepcopy(theta)
            
            for l in range(nLocalSteps):
                # sigma = sigma0 / (t + 1)**0.5 / (l + 1)**0.5
                
                # 生成随机扰动方向
                v = np.random.normal(size=sizes)
                v_rand = v * sigma
                # 随机选择小批次
                minibatchidx = np.random.choice(Nis[i], min(batchsize, Nis[i]), replace=False)
                local_fobj = lambda w: model.evaluate_func(X_trains[i][minibatchidx, :], Y_trains[i][minibatchidx], w)
                
                # 零阶梯度估计
                pre_local_fobj = local_fobj(local_solution - v_rand)
                current_local_fobj = local_fobj(local_solution + v_rand)
                g_trial = v * (current_local_fobj - pre_local_fobj) / (2 * mu)
                
                # signSGD更新 - 只使用梯度的符号
                local_solution = local_solution - np.sign(g_trial) * sigma
            
            worker_solutions[i, :] = local_solution
        
        # 全局更新
        theta = np.mean(worker_solutions, axis=0)
        
        # 计算性能指标
        train_loss = model.loss(X_train, Y_train, theta)
        predict = model.predict(X_test, theta)
        test_acc = model.score(predict, Y_test)
        
        loss_.append(train_loss)
        accuracy_.append(test_acc)
        
        print(f"round {t+1}, training loss = {round(train_loss, 6)}, test accuracy = {round(test_acc, 6)}")
    
    return loss_, accuracy_

def run_fed_zo_sgd_cuda(X_train, Y_train, X_test, Y_test, Ntrain, n, nWorkers,
                   batchsize, data_idxs, nLocalSteps, global_epoch, sigma0, theta, model,
                   client_lr=0.01, server_lr=1.0, device='cuda'):
    """
    Fed-ZO-SGD: 联邦零阶SGD算法 - CUDA版本
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Running Fed-ZO-SGD on {device}...")
    
    # 将数据移动到设备
    X_train = move_to_device(X_train, device)
    Y_train = move_to_device(Y_train, device)
    X_test = move_to_device(X_test, device)
    Y_test = move_to_device(Y_test, device)
    
    loss_, accuracy_ = [], []
    server_lr = 0.1
    # 为每个工作节点准备数据
    X_trains, Y_trains, Nis = [], [], []
    for i in range(nWorkers):
        Nis.append(len(data_idxs[i]))
        X_trains.append(X_train[data_idxs[i], :])
        Y_trains.append(Y_train[data_idxs[i]])
    sizes = theta.shape
    for t in range(global_epoch):
        if t == 0:
            train_loss = model.loss(X_train, Y_train, theta)
            predict = model.predict(X_test, theta)
            test_acc = model.score(predict, Y_test)
            loss_.append(train_loss)
            accuracy_.append(test_acc)
            print(f"round {0}, training loss = {round(train_loss, 6)}, test accuracy = {round(test_acc, 6)}")
        
        worker_updates = np.zeros((nWorkers, n))
        mu = 1e-3  # 高斯平滑参数
        
        # 客户端更新
        for i in range(nWorkers):
            local_theta = deepcopy(theta)
            
            for l in range(nLocalSteps):
                # 客户端步长
                client_sigma = 0.001
                
                # 生成随机扰动方向
                v = np.random.normal(size=sizes)
                v_rand = v * mu

                
                # 随机选择小批次
                minibatchidx = np.random.choice(Nis[i], min(batchsize, Nis[i]), replace=False)
                local_fobj = lambda w: model.evaluate_func(X_trains[i][minibatchidx, :], Y_trains[i][minibatchidx], w)
                
                # 零阶梯度估计
                pre_local_fobj = local_fobj(local_theta - v_rand)
                current_local_fobj = local_fobj(local_theta + v_rand)
                g_trial = v * (current_local_fobj - pre_local_fobj) / (2 * mu)
                
                # 客户端参数更新
                local_theta = local_theta - client_sigma * g_trial
            
            worker_updates[i, :] = local_theta - theta  # 记录参数更新
        
        # 服务器聚合 - 使用服务器学习率
        theta = theta + server_lr * np.mean(worker_updates, axis=0)
        
        # 计算性能指标
        train_loss = model.loss(X_train, Y_train, theta)
        predict = model.predict(X_test, theta)
        test_acc = model.score(predict, Y_test)
        
        loss_.append(train_loss)
        accuracy_.append(test_acc)
        
        print(f"round {t+1}, training loss = {round(train_loss, 6)}, test accuracy = {round(test_acc, 6)}")
    
    return loss_, accuracy_