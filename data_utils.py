import numpy as np
import pickle
import gzip
import urllib.request
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataLoader:
    """数据加载器基类"""
    
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_val = None
        self.y_val = None
    
    def load_data(self):
        """加载数据，子类需要实现"""
        raise NotImplementedError
    
    def normalize_data(self, X_train, X_test):
        """数据标准化"""
        # 将像素值归一化到[0, 1]范围
        X_train = X_train.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0
        
        return X_train, X_test
    
    def create_validation_set(self, X_train, y_train, val_size=0.1, random_state=42):
        """创建验证集"""
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train
        )
        return X_train, X_val, y_train, y_val
    
    def get_data(self):
        """获取数据"""
        return self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val
    
    def get_info(self):
        """获取数据集信息"""
        info = {
            'train_samples': len(self.X_train) if self.X_train is not None else 0,
            'test_samples': len(self.X_test) if self.X_test is not None else 0,
            'val_samples': len(self.X_val) if self.X_val is not None else 0,
            'input_dim': self.X_train.shape[1] if self.X_train is not None else 0,
            'num_classes': len(np.unique(self.y_train)) if self.y_train is not None else 0
        }
        return info
# https://yann.lecun.org/exdb/mnist/train-images-idx3-ubyte.gz
class MNISTLoader(DataLoader):
    """MNIST数据集加载器"""
    
    def __init__(self, data_dir='data'):
        super().__init__()
        self.data_dir = data_dir
        self.url = 'https://yann.lecun.org/exdb/mnist/'
        self.files = {
            'train_images': 'train-images-idx3-ubyte.gz',
            'train_labels': 'train-labels-idx1-ubyte.gz',
            'test_images': 't10k-images-idx3-ubyte.gz',
            'test_labels': 't10k-labels-idx1-ubyte.gz'
        }
    
    def download_file(self, filename):
        """下载文件"""
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            os.makedirs(self.data_dir, exist_ok=True)
            urllib.request.urlretrieve(self.url + filename, filepath)
            print(f"Downloaded {filename}")
        return filepath
    
    def load_images(self, filepath):
        """加载图像数据"""
        with gzip.open(filepath, 'rb') as f:
            # 读取magic number和维度信息
            magic = int.from_bytes(f.read(4), 'big')
            num_images = int.from_bytes(f.read(4), 'big')
            rows = int.from_bytes(f.read(4), 'big')
            cols = int.from_bytes(f.read(4), 'big')
            
            # 读取图像数据
            image_data = f.read()
            images = np.frombuffer(image_data, dtype=np.uint8)
            images = images.reshape(num_images, rows * cols)
            
            return images
    
    def load_labels(self, filepath):
        """加载标签数据"""
        with gzip.open(filepath, 'rb') as f:
            # 读取magic number和数量信息
            magic = int.from_bytes(f.read(4), 'big')
            num_labels = int.from_bytes(f.read(4), 'big')
            
            # 读取标签数据
            label_data = f.read()
            labels = np.frombuffer(label_data, dtype=np.uint8)
            
            return labels
    
    def load_data(self, val_size=0.1, create_val=True):
        """加载MNIST数据"""
        print("Loading MNIST dataset...")
        
        # 下载并加载训练数据
        train_images_path = './data/MNIST/raw/train-images-idx3-ubyte.gz'
        # train_images_path = self.download_file(self.files['train_images'])
        # train_labels_path = self.download_file(self.files['train_labels'])
        train_labels_path = './data/MNIST/raw/train-labels-idx1-ubyte.gz'
        X_train = self.load_images(train_images_path)
        y_train = self.load_labels(train_labels_path)
        
        # 下载并加载测试数据
        # test_images_path = self.download_file(self.files['test_images'])
        # test_labels_path = self.download_file(self.files['test_labels'])
        test_images_path = './data/MNIST/raw/t10k-images-idx3-ubyte.gz'
        test_labels_path = './data/MNIST/raw/t10k-labels-idx1-ubyte.gz'
        
        X_test = self.load_images(test_images_path)
        y_test = self.load_labels(test_labels_path)
        
        # 数据标准化
        X_train, X_test = self.normalize_data(X_train, X_test)
        
        # 创建验证集
        if create_val:
            X_train, X_val, y_train, y_val = self.create_validation_set(X_train, y_train, val_size)
            self.X_val = X_val
            self.y_val = y_val
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        print(f"MNIST dataset loaded successfully!")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val) if create_val else 0}")
        print(f"Test samples: {len(X_test)}")
        print(f"Input dimension: {X_train.shape[1]}")
        print(f"Number of classes: {len(np.unique(y_train))}")
        
        return self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val

class CIFAR10Loader(DataLoader):
    """CIFAR-10数据集加载器"""
    
    def __init__(self, data_dir='data'):
        super().__init__()
        self.data_dir = data_dir
        self.url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        self.tar_file = 'cifar-10-python.tar.gz'
    
    def download_and_extract(self):
        """下载并解压CIFAR-10数据"""
        tar_path = os.path.join(self.data_dir, self.tar_file)
        extract_dir = os.path.join(self.data_dir, 'cifar-10-batches-py')
        
        if not os.path.exists(extract_dir):
            print("Downloading CIFAR-10 dataset...")
            os.makedirs(self.data_dir, exist_ok=True)
            urllib.request.urlretrieve(self.url, tar_path)
            
            # 解压文件
            import tarfile
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(self.data_dir)
            print("CIFAR-10 dataset downloaded and extracted!")
        
        return extract_dir
    
    def unpickle(self, file):
        """解压pickle文件"""
        with open(file, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        return data
    
    def load_data(self, val_size=0.1, create_val=True):
        """加载CIFAR-10数据"""
        print("Loading CIFAR-10 dataset...")
        
        extract_dir = self.download_and_extract()
        
        # 加载训练批次
        train_batches = []
        train_labels = []
        
        for i in range(1, 6):
            batch_file = os.path.join(extract_dir, f'data_batch_{i}')
            batch_data = self.unpickle(batch_file)
            
            train_batches.append(batch_data[b'data'])
            train_labels.extend(batch_data[b'labels'])
        
        X_train = np.vstack(train_batches)
        y_train = np.array(train_labels)
        
        # 加载测试批次
        test_file = os.path.join(extract_dir, 'test_batch')
        test_data = self.unpickle(test_file)
        
        X_test = test_data[b'data']
        y_test = np.array(test_data[b'labels'])
        
        # 数据标准化
        X_train, X_test = self.normalize_data(X_train, X_test)
        
        # 创建验证集
        if create_val:
            X_train, X_val, y_train, y_val = self.create_validation_set(X_train, y_train, val_size)
            self.X_val = X_val
            self.y_val = y_val
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        print(f"CIFAR-10 dataset loaded successfully!")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val) if create_val else 0}")
        print(f"Test samples: {len(X_test)}")
        print(f"Input dimension: {X_train.shape[1]}")
        print(f"Number of classes: {len(np.unique(y_train))}")
        
        return self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val

def create_data_loaders(dataset_name='mnist', data_dir='data'):
    """
    创建数据加载器
    
    参数:
    dataset_name: 数据集名称 ('mnist' 或 'cifar10')
    data_dir: 数据存储目录
    
    返回:
    数据加载器对象
    """
    if dataset_name.lower() == 'mnist':
        return MNISTLoader(data_dir)
    elif dataset_name.lower() == 'cifar10':
        return CIFAR10Loader(data_dir)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def prepare_distributed_data(X_train, y_train, n_workers, balanced=True, random_state=42):
    """
    为分布式训练准备数据
    
    参数:
    X_train: 训练数据
    y_train: 训练标签
    n_workers: 工作节点数量
    balanced: 是否平衡分配数据
    random_state: 随机种子
    
    返回:
    data_idxs: 每个工作节点的数据索引列表
    """
    n_samples = len(X_train)
    data_idxs = []
    
    if balanced:
        # 平衡分配 - 每个工作节点获得相同数量的每个类别的样本
        unique_labels = np.unique(y_train)
        n_classes = len(unique_labels)
        
        # 为每个类别分配样本给工作节点
        class_indices = {label: np.where(y_train == label)[0] for label in unique_labels}
        
        for worker_id in range(n_workers):
            worker_indices = []
            for label in unique_labels:
                class_idx = class_indices[label]
                n_samples_per_class = len(class_idx) // n_workers
                start_idx = worker_id * n_samples_per_class
                end_idx = (worker_id + 1) * n_samples_per_class if worker_id < n_workers - 1 else len(class_idx)
                worker_indices.extend(class_idx[start_idx:end_idx])
            
            data_idxs.append(np.array(worker_indices))
    else:
        # 随机分配
        indices = np.arange(n_samples)
        np.random.seed(random_state)
        np.random.shuffle(indices)
        
        samples_per_worker = n_samples // n_workers
        for worker_id in range(n_workers):
            start_idx = worker_id * samples_per_worker
            end_idx = (worker_id + 1) * samples_per_worker if worker_id < n_workers - 1 else n_samples
            data_idxs.append(indices[start_idx:end_idx])
    
    return data_idxs

if __name__ == "__main__":
    # 测试数据加载器
    print("Testing MNIST loader...")
    mnist_loader = MNISTLoader()
    X_train, y_train, X_test, y_test, X_val, y_val = mnist_loader.load_data()
    
    print("\nTesting CIFAR-10 loader...")
    cifar_loader = CIFAR10Loader()
    X_train, y_train, X_test, y_test, X_val, y_val = cifar_loader.load_data()
    
    # 测试分布式数据准备
    print("\nTesting distributed data preparation...")
    data_idxs = prepare_distributed_data(X_train, y_train, n_workers=5)
    print(f"Created data indices for {len(data_idxs)} workers")
    for i, idx in enumerate(data_idxs):
        print(f"Worker {i}: {len(idx)} samples")