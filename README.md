# DZO-SNGM: Momentum-based Zeroth-order Gradient Method for Distributed Black-box Optimization

This is the official implementation of **“Momentum-based Zeroth-order Gradient Method for Distributed Black-box Optimization.”**

---

## Project Overview

**DZO-SNGM** (Distributed Zeroth-Order Stochastic Natural Gradient Momentum) is a zeroth-order optimization framework designed for distributed machine learning.
This project provides a complete implementation of the proposed DZO-SNGM algorithm.

---

## Key Features

* **Multiple algorithm implementations**: Includes 7 distributed zeroth-order optimization algorithms
* **CUDA acceleration**: Supports GPU acceleration for significantly faster training
* **Multi-dataset support**: Works with both MNIST and CIFAR-10 datasets
* **Distributed training**: Supports multi-worker distributed training environments
* **Adaptive parameters**: Includes various adaptive learning rate and step-size strategies
* **Complete experimental framework**: Covers data loading, model definition, training, and result analysis

---

## Project Structure

```
DZO-SNGM/
├── code/                    # Core code directory
│   ├── main.py              # Main entry point
│   ├── IntegratedZO.py      # Integrated optimization framework
│   ├── data_utils.py        # Data loading and preprocessing utilities
│   ├── model_cuda.py        # CUDA version of models
│   └── algorithms_cuda.py   # CUDA implementations of optimization algorithms
├── data/                    # Dataset storage
│   ├── MNIST/               # MNIST dataset
│   └── cifar-10-batches-py/ # CIFAR-10 dataset
```

---

## Installation Requirements

### Basic Dependencies

```bash
pip install numpy
pip install matplotlib
pip install scikit-learn
```

### Optional: CUDA Support

```bash
pip install torch
```

---

## Quick Start

### 1. Basic Usage

Run DZO-SNGM on the MNIST dataset:

```bash
cd DZO-SNGM/code
python main.py --dataset mnist --method DZO-SNGM
```

### 2. With CUDA Acceleration

```bash
python main.py --dataset mnist --method DZO-SNGM --use_cuda --device cuda
```

### 3. Run All Algorithms for Comparison

```bash
python main.py --dataset mnist --compare-all
```

### 4. Train on CIFAR-10

```bash
python main.py --dataset cifar10 --method DZO-SNGM --use_cuda
```

### 5. Train on Both Datasets

```bash
python main.py --dataset both --method DZO-SNGM --use_cuda
```

---

## Command-line Arguments

```bash
python main.py [options]

Options:
  --dataset {mnist,cifar10,both}   Dataset selection (default: mnist)
  --method METHOD_NAME             Optimization method (default: DZO-SNGM)
  --batchsize INT                  Batch size (default: 256)
  --compare-all                    Run all available methods for comparison
  --use_cuda                       Enable CUDA acceleration (if available)
  --device {cuda,cpu}              Device selection (default: cuda)
  -h, --help                       Show help message
```

---

## Algorithm Configuration

Each algorithm has its own hyperparameter configuration, accessible through the `get_method_config()` function.

### DZO-SNGM Configuration

* `beta1`: 0.9 (first-moment decay coefficient)
* `beta2`: 0.999 (second-moment decay coefficient)
* `epsilon`: 1e-8 (numerical stability constant)
* `momentum_factor`: 0.5 (momentum term)
* `adaptive_sigma`: True (adaptive smoothing parameter)

### Other Algorithms

Default configurations for other methods can be found and modified in `IntegratedZO.py`.

---

## Experimental Setup

### Default Training Configuration

* **Number of workers**: 10
* **Local update steps**: 10
* **Global iterations**: 1000 (MNIST) / 2000 (CIFAR-10)
* **Batch size**: 256
* **Initial step size**: 1e-3
* **Gaussian smoothing parameter**: 1e-3

### Data Distribution Strategies

1. **Balanced distribution**: Each worker receives an equal number of samples per class
2. **Random distribution**: Data are randomly assigned to workers

---

## Result Analysis

### Output Files

Training results are automatically saved in the `results/` directory, including:

* Training loss history
* Test accuracy history
* Configuration parameters
* Final performance metrics


## Technical Details

### Zeroth-Order Gradient Estimation

The symmetric finite-difference estimator is used:

```
g_trial = v * (f(w + v*μ) - f(w - v*μ)) / (2*μ)
```

## Extension and Customization

### Adding a New Algorithm

1. Implement the new method in `algorithms.py`
2. Add a CUDA version in `algorithms_cuda.py`
3. Register the method in `IntegratedZO.py`
4. Update command-line options accordingly

### Supporting a New Dataset

1. Create a new data loader in `data_utils.py`
2. Implement the preprocessing pipeline
3. Define an appropriate model in `model.py`

### Custom Models

Custom architectures can be added by modifying the `SimpleDNN` class in `model.py` or defining new model classes.

---

## Notes and Recommendations

1. **Memory Usage**: CUDA versions require sufficient GPU memory; large batch sizes may cause OOM errors.
2. **Random Seeds**: Fixed seeds are used for reproducibility.
3. **Numerical Stability**: Built-in mechanisms prevent issues like division by zero.
4. **Convergence Behavior**: Performance may vary across datasets and algorithm variants.

---

## Citation

If you use DZO-SNGM in your research, please cite:

```bibtex
To be released
```

---

## License

This project is released under the **MIT License**. See the `LICENSE` file for details.

---

## Contact

For questions or suggestions, please contact:

* Email: **[xidianqldang@163.com](mailto:xidianqldang@163.com)**
* GitHub Issues: Submit an issue report

---

## Changelog

### v1.0.0 (October 2025)

* Initial public release
* Implemented 7 distributed zeroth-order optimization algorithms
* Added support for MNIST and CIFAR-10 datasets
* Integrated CUDA acceleration
* Complete experimental and visualization framework
