import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDNN_CUDA(nn.Module):
    """
    支持CUDA的简单深度神经网络模型，用于多分类任务
    支持MNIST和CIFAR-10数据集
    """
    
    def __init__(self, input_dim, hidden_dims, output_dim, activation='relu', device='cuda'):
        """
        初始化神经网络
        
        参数:
        input_dim: 输入维度
        hidden_dims: 隐藏层维度列表，例如 [128, 64]
        output_dim: 输出维度（类别数）
        activation: 激活函数类型 ('relu' 或 'tanh')
        device: 设备类型 ('cuda' 或 'cpu')
        """
        super(SimpleDNN_CUDA, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation_type = activation
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 构建网络结构
        self.layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layer = nn.Linear(dims[i], dims[i+1])
            # Xavier初始化
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.layers.append(layer)
        
        # 将模型移动到设备
        self.layers = self.layers.to(self.device)
        
        # 获取参数数量
        self.param_count = sum(p.numel() for p in self.parameters())
        
        # 缓存参数向量
        self._update_param_vector()
    
    def _update_param_vector(self):
        """更新扁平化的参数向量"""
        params = []
        for layer in self.layers:
            params.append(layer.weight.data.flatten())
            params.append(layer.bias.data.flatten())
        self.theta = torch.cat(params).cpu().numpy()
    
    def set_params(self, theta):
        """
        设置模型参数
        
        参数:
        theta: 扁平化的参数向量 (numpy数组)
        """
        theta_tensor = torch.FloatTensor(theta).to(self.device)
        start_idx = 0
        
        for layer in self.layers:
            weight_size = layer.weight.numel()
            bias_size = layer.bias.numel()
            
            # 提取权重和偏置
            weight_data = theta_tensor[start_idx:start_idx + weight_size].reshape(layer.weight.shape)
            bias_data = theta_tensor[start_idx + weight_size:start_idx + weight_size + bias_size].reshape(layer.bias.shape)
            
            layer.weight.data = weight_data
            layer.bias.data = bias_data
            
            start_idx += weight_size + bias_size
        
        self.theta = theta.copy()
    
    def get_params(self):
        """获取扁平化的参数向量"""
        self._update_param_vector()
        return self.theta.copy()
    
    def forward(self, X):
        """
        前向传播
        
        参数:
        X: 输入数据，形状 (n_samples, input_dim)
        
        返回:
        输出概率，形状 (n_samples, output_dim)
        """
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X).to(self.device)
        
        current_input = X
        
        # 前向传播通过隐藏层
        for i, layer in enumerate(self.layers[:-1]):
            current_input = layer(current_input)
            if self.activation_type == 'relu':
                current_input = F.relu(current_input)
            elif self.activation_type == 'tanh':
                current_input = torch.tanh(current_input)
        
        # 输出层
        output = self.layers[-1](current_input)
        
        # 应用softmax得到概率
        probs = F.softmax(output, dim=1)
        
        return probs
    
    def predict(self, X, theta=None):
        """
        预测函数 - 用于ZO优化算法
        
        参数:
        X: 输入数据
        theta: 模型参数（可选）
        
        返回:
        预测类别 (numpy数组)
        """
        if theta is not None:
            self.set_params(theta)
        
        with torch.no_grad():
            probs = self.forward(X)
            predictions = torch.argmax(probs, dim=1).cpu().numpy()
        
        return predictions
    
    def loss(self, X, y, theta=None):
        """
        计算损失函数 - 交叉熵损失
        
        参数:
        X: 输入数据
        y: 真实标签
        theta: 模型参数（可选）
        
        返回:
        平均损失值 (float)
        """
        if theta is not None:
            self.set_params(theta)
        
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X).to(self.device)
        if isinstance(y, np.ndarray):
            y = torch.LongTensor(y).to(self.device)
        
        with torch.no_grad():
            probs = self.forward(X)
            
            # 交叉熵损失
            if len(y.shape) == 1:  # 类别标签
                # 确保y是long类型
                y_long = y.long()
                loss = F.cross_entropy(torch.log(probs + 1e-15), y_long)
            else:  # one-hot编码
                loss = -torch.sum(y * torch.log(probs + 1e-15)) / X.shape[0]
            
            return loss.item()
    
    def score(self, y_pred, y_true):
        """
        计算准确率
        
        参数:
        y_pred: 预测标签
        y_true: 真实标签
        
        返回:
        准确率
        """
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        return np.mean(y_pred == y_true)
    
    def evaluate_func(self, X, y, theta):
        """
        评估函数 - 用于ZO优化算法
        
        参数:
        X: 输入数据
        y: 真实标签
        theta: 模型参数
        
        返回:
        损失值
        """
        return self.loss(X, y, theta)
    
    def get_param_count(self):
        """获取参数总数"""
        return self.param_count
    
    def summary(self):
        """打印模型结构摘要"""
        print("Simple DNN CUDA Model Summary:")
        print(f"Input dimension: {self.input_dim}")
        print(f"Hidden layers: {self.hidden_dims}")
        print(f"Output dimension: {self.output_dim}")
        print(f"Activation function: {self.activation_type}")
        print(f"Device: {self.device}")
        print(f"Total parameters: {self.get_param_count()}")
        
        for i, layer in enumerate(self.layers):
            print(f"Layer {i+1}: {layer.in_features} -> {layer.out_features}")

# 预定义模型配置
def get_mnist_model_cuda():
    """获取MNIST数据集对应的CUDA模型"""
    return SimpleDNN_CUDA(input_dim=784, hidden_dims=[128, 64], output_dim=10, activation='relu')

def get_cifar10_model_cuda():
    """获取CIFAR-10数据集对应的CUDA模型"""
    return SimpleDNN_CUDA(input_dim=3072, hidden_dims=[256, 128, 64], output_dim=10, activation='relu')

def get_toy_model_cuda(input_dim=20, output_dim=2):
    """获取简单的测试CUDA模型"""
    return SimpleDNN_CUDA(input_dim=input_dim, hidden_dims=[32, 16], output_dim=output_dim, activation='relu')