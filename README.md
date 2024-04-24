# Variational Continual Learning

This repository contains the implementation for the mini-project **Variational Continual Learning**.

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- Tqdm
- Scienceplots

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/your-username/variational-continual-learning.git
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the `permuted_mnist.py` (VCL), `permuted_mnist_no_data.py` (implementation of SVCL)  or `split_mnist.py` (VCL) with the desired configuration options:

```bash
python permuted_mnist.py --epochs 50 --nb_simul 10 --batch_size 1024 --lr 0.001 --hidden_size 100 --beta 1 --vcl --reduction mean --num_tasks 10 --method reparametrization --coreset_size 200 --train_ratio 0.8 --filename "example_output"
```

- `--epochs`: Number of epochs per task (default: 20).
- `--nb_simul`: Number of simulations (default: 5).
- `--batch_size`: Batch size for training (default: 1024).
- `--lr`: Learning rate (default: 0.001).
- `--hidden_size`: Hidden size of the Bayesian model (default: 100).
- `--beta`: Beta parameter for VCL (default: 1).
- `--vcl`: Use Variational Continual Learning (default: False).
- `--reduction`: Reduction method for the loss (default: 'mean').
- `--num_tasks`: Number of tasks (default: 10). (Only for PermutedMNIST)
- `--method`: Method of forwarding, local parametrization or not (default: 'reparametrization').
- `--coreset_size`: Size of the coreset (default: 0).
- `--train_ratio`: Ratio of data to use for training (default: 0.8).
- `--filename`: Name of the output PDF file (default: '').
- `--bias`: Use bias (default: False).
- `--recompute_data`: Recompute the Permuted MNIST data, otherwise a saved version is used (default: False).
- `--compute_baseline`: Compute the baseline accuracy (default: False).
- `--seed`: Random seed.

## Outputs

The outputs include trained models, logs of accuracy and losses, and plots visualizing the performance of the model over tasks.

## References

- [Variational Continual Learning](https://arxiv.org/pdf/1710.10628.pdf)
- [Improving and Understanding Variational Continual Learning](https://arxiv.org/pdf/1905.02099.pdf)
```

















TO DELETE

To test:
- parametrize the betas with scheduler!
- how to force first layers to unpermute
- highly sensible to intialization and all, so what about trying to find a better starting point
- pb is high variance of bayesian task, so that kl div optimization is much "easier", hence choosing smaller beta improves, try scheduling as decreasing