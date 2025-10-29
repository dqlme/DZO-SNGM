#!/usr/bin/env python3
"""
测试集成ZO算法的脚本
"""

import numpy as np
import os
import sys
import json
import matplotlib.pyplot as plt
import argparse
# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import get_mnist_model, get_cifar10_model, get_toy_model
from model_cuda import get_mnist_model_cuda, get_cifar10_model_cuda, get_toy_model_cuda
from data_utils import create_data_loaders, prepare_distributed_data

# 动态导入集成ZO模块
import importlib.util
spec = importlib.util.spec_from_file_location("IntegratedZO",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),  'IntegratedZO.py'))
IntegratedZO_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(IntegratedZO_module)

IntegratedZO = IntegratedZO_module.IntegratedZO
run_integrated_zo = IntegratedZO_module.run_integrated_zo
get_method_config = IntegratedZO_module.get_method_config

import torch
import torch.nn as nn
import numpy as np
import random
import os

def set_seed(seed=42):
    """
    设置全局随机种子以确保实验可复现
    """
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化
    random.seed(seed)                         # Python random 模块
    np.random.seed(seed)                      # NumPy
    torch.manual_seed(seed)                   # CPU 随机种子
    torch.cuda.manual_seed(seed)              # 当前 GPU
    torch.cuda.manual_seed_all(seed)          # 所有 GPU
    torch.backends.cudnn.deterministic = True # 使用确定性算法
    torch.backends.cudnn.benchmark = False    # 禁用 CuDNN 的自动优化（否则会牺牲可复现性）
    torch.backends.cudnn.enabled = False      # 可选：完全禁用 CuDNN（更严格）


def test_real_dataset(dataset_name='mnist', method_name='DZO-SNGM', batchsize=256, use_cuda=False, device='cuda'):
    """在真实数据集上测试 - 支持CUDA"""
    print(f"\nTesting {method_name} on {dataset_name} dataset...")
    print(f"Device: {'CUDA' if use_cuda else 'CPU'}")
    
    np.random.seed(42)
    set_seed(42)
    # 加载数据
    data_loader = create_data_loaders(dataset_name)
    X_train, y_train, X_test, y_test, X_val, y_val = data_loader.load_data()
    
    # 创建模型
    if dataset_name == 'mnist':
        if use_cuda:
            model = get_mnist_model_cuda()
        else:
            model = get_mnist_model()
    elif dataset_name == 'cifar10':
        if use_cuda:
            model = get_cifar10_model_cuda()
        else:
            model = get_cifar10_model()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # 准备分布式数据
    data_idxs = prepare_distributed_data(X_train, y_train, n_workers=10, balanced=True)
    
    # 获取初始参数
    theta = model.get_params()
    n_params = len(theta)
    
    print(f"Dataset: {dataset_name}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Model parameters: {n_params}")
    
    # 运行优化
    start_time = time.time()
    
    results = run_integrated_zo(
        method=method_name,
        X_train=X_train,
        Y_train=y_train,
        X_test=X_test,
        Y_test=y_test,
        Ntrain=len(X_train),
        n=n_params,
        nWorkers=10,
        batchsize=batchsize,
        data_idxs=data_idxs,
        nLocalSteps=10,
        global_epoch=1000 if dataset_name == 'mnist' else 2000,
        sigma0=1,
        theta=theta,
        model=model,
        dataset=dataset_name,
        use_cuda=use_cuda,
        device=device
    )
    
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Final accuracy: {results['acc_history'][-1]:.4f}")
    print(f"Final loss: {results['loss_history'][-1]:.6f}")
    
    # 保存结果
    os.makedirs('results', exist_ok=True)
    import datetime
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")    
    # 保存到文本文件
    device_str = 'cuda' if use_cuda else 'cpu'
    history_file = f"results/{dataset_name}_{method_name}_{device_str}_{current_time}.txt"
    with open(history_file, 'w') as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Method: {method_name}\n")
        f.write(f"Device: {device_str}\n")
        f.write(f"Training Configuration:\n")
        f.write(f"  Workers: 10\n")
        f.write(f"  Batch size: {256}\n")
        f.write(f"  Local steps: {10}\n")
        f.write(f"  Global epochs: {1000 if dataset_name == 'mnist' else 2000}\n")
        f.write(f"  sigma: {1e-3}\n")
        f.write(f"  sigma: {1e-3}\n")
        f.write(f"  beta: {0.9}\n")
        f.write(f"\nTraining History:\n")
        f.write("Epoch\tLoss\t\tAccuracy\n")
        f.write("-" * 30 + "\n")
        
        for i, (loss, acc) in enumerate(zip(results['loss_history'], results['acc_history'])):
            f.write(f"{i+1}\t{loss:.6f}\t{acc:.4f}\n")
        
        f.write("\nFinal Results:\n")
        f.write(f"Final Loss: {results['loss_history'][-1]:.6f}\n")
        f.write(f"Final Accuracy: {results['acc_history'][-1]:.4f}\n")
        f.write(f"Training Time: {training_time:.2f} seconds\n")
    
    print(f"Results saved to {history_file}")
    
    return results

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='ZO-SGD Multi-class Classification')
    
    # 数据集选择
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'cifar10', 'both'],
                       help='Dataset to use (default: mnist)')
    parser.add_argument('--method', type=str, default='DZO-SNGM',
                       choices=['DZO-SNGM','DZO-SGD', 'DZO-Adam', 'DZO-SVRG','DES','DZO-signSGD','Fed-ZO-SGD'],
                       help='Method to use (default: DZO-SNGM)')
    parser.add_argument('--batchsize',type=int, default=256,
                       help='Test first round consistency across all methods')
    parser.add_argument('--compare-all', action='store_true',
                       help='Compare all methods')
    parser.add_argument('--use_cuda', action='store_true',
                       help='Use CUDA acceleration if available')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for training (default: cuda)')
    return parser.parse_args()

if __name__ == "__main__":
    import time
    
    print("=" * 60)
    print("Integrated ZO Algorithms Test")
    print("=" * 60)
    
    # # 1. 快速合成数据测试
    # print("\n1. Testing on synthetic data...")
    # results = compare_all_methods()
    # plot_comparison(results)
    args = parse_arguments()

    # 单个方法测试
    if args.dataset == 'both':
        # 2. MNIST数据集测试
        print("\n2. Testing on MNIST dataset...")
        mnist_results = test_real_dataset('mnist', args.method, args.use_cuda, args.device)
        
        # 3. CIFAR-10数据集测试
        print("\n3. Testing on CIFAR-10 dataset...")
        cifar_results = test_real_dataset('cifar10', args.method, args.use_cuda, args.device)
    else:
        print(f"\n2. Testing on {args.dataset.upper()} dataset...")
        results = test_real_dataset(args.dataset, args.method, args.batchsize, args.use_cuda, args.device)
    
    # # 3. CIFAR-10数据集测试  
    # print("\n3. Testing on CIFAR-10 dataset...")
    # cifar_results = test_real_dataset('cifar10', 'DZO-SNGM')
    
    # print("\n" + "=" * 60)
    # print("All tests completed!")
    # print("=" * 60)