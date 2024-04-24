import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime
from tqdm import tqdm
import pickle
import scienceplots
plt.style.use('science')

from models.Bayesian.bayesianmlp import BayesianNN
from utils import compute_kl_divergence, create_coresets, set_seed
from data_creation.permuted_data import load_permuted_mnist



def continual_learning(model, model2, train_datasets, coreset_datasets, test_datasets, tasks, epochs_per_task=5, batch_size=64, lr=0.001, nb_simul=10, vcl=False, reduction='sum', beta=1, beta_scheduler=None, compute_baseline=False, reuse_first=False, name=None):
    criterion = nn.CrossEntropyLoss(reduction=reduction)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        print("Using GPU, but never tested before...")
    model.to(device)
    
    if beta_scheduler is None:
        beta_scheduler = [beta]*epochs_per_task
    
    # Train the model on the union of all train datasets
    if compute_baseline:
        all_train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        all_train_loader = DataLoader(all_train_dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        for epoch in tqdm(range(epochs_per_task), desc='Computing baseline.'):
            model.train()
            running_loss = 0.0
            for images, labels in all_train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            epoch_loss = running_loss / len(all_train_loader)
            print(f'Epoch [{epoch+1}/{epochs_per_task}], Loss: {epoch_loss:.4f}') # Integrate in tqdm

        # Evaluate the model on the union of all test datasets
        overall_baseline, _ = evaluate_model(model, test_datasets, nb_simul, device, batch_size=batch_size)
        print(f'Overall Accuracy on union of test sets: {overall_baseline:.4f}')
    else:
        overall_baseline = None

    # Main VCL
    
    model.reset_parameters()
    mu_current, logvar_current = model.get_prior_parameters()
    mu_previous, logvar_previous = mu_current.detach(), logvar_current.detach()
    mu_previous0, logvar_previous0 = mu_current.detach(), logvar_current.detach()
    model.train()
    overall_accuracies = []
    task_accuracies = []
    losses = []
    mu_previous_list = []
    logvar_previous_list = []
    
    
    for task_id, (task_start, task_end) in enumerate(tasks):
        kl_losses = np.array([0.]*epochs_per_task)
        kl_losses2 = np.array([0.]*epochs_per_task)
        kl_losses0 = np.array([0.]*epochs_per_task)
        log_losses = np.array([0.]*epochs_per_task)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        if reuse_first and task_id == 0:
            model.load_state_dict(torch.load(f'./outputs/PermutedMNIST/models/model_{0}.pt'))
            mu_current, logvar_current = model.get_variational_parameters()
            mu_previous, logvar_previous = mu_current.detach(), logvar_current.detach()
            continue
        concatenated_train_dataset = ConcatDataset([train_datasets[task_id]] + coreset_datasets[:task_id]) # coreset not yet fully implemented, evaluation missing
        train_loader = DataLoader(concatenated_train_dataset, batch_size=batch_size, shuffle=True)
        len_train_loader = len(train_loader)
        # print(len_train_loader, len(train_datasets[task_id]))
        task_accuracies.append([])
        task_losses = []
            
        # Learn the local model
        weights = model.get_weights()
        model2.load_weights(weights)
        model2.train()
        optimizer2 = optim.Adam(model2.parameters(), lr=lr)
        for epoch in tqdm(range(epochs_per_task), desc=f'InitTask {task_id+1}'):
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                for _ in range(nb_simul):
                    optimizer2.zero_grad()
                    outputs = model2(images)
                    loss = torch.tensor(0.0)
                    log_likelihood_loss = -torch.log_softmax(outputs, dim=1)[range(labels.size(0)), labels].mean()
                    loss += log_likelihood_loss
                    loss.backward()
                    optimizer2.step()
        # eval
        overall_accuracy, task_accuracy = evaluate_model(model2, test_datasets[:task_id+1], nb_simul, device, batch_size=batch_size)
        print(f"InitTask {task_id+1}: Overall accuracy = {overall_accuracy:.4f}, Task accuracies = {task_accuracy}")
        mu_current2, logvar_current2 = model2.get_variational_parameters()
        mu_previous2, logvar_previous2 = mu_current2.detach(), logvar_current2.detach()
        mu_previous_list.append(mu_previous2)
        logvar_previous_list.append(logvar_previous2)
        
        # weights2 = model2.get_weights()
        # model.load_weights(weights2)
        
        model.reset_parameters() # should i let it???
        # weights = model.get_weights()
        # weights2 = model2.get_weights()
        # weights_average = [torch.nn.Parameter((weights[i]+weights2[i])/2) for i in range(len(weights))]
        # model.load_weights(weights_average)
        
        
        
        # eval
        # overall_accuracy, task_accuracy = evaluate_model(model, test_datasets[:task_id+1], nb_simul, device, batch_size=batch_size)
        # print(f"Average Task {task_id+1}: Overall accuracy = {overall_accuracy:.4f}, Task accuracies = {task_accuracy}")
        # model.reset_parameters()
        for epoch in tqdm(range(100), desc=f'Task {task_id+1}'):
            running_loss = 0.0
            for _ in range(nb_simul):
                optimizer.zero_grad()
                outputs = model(images)
                loss = torch.tensor(0.0)
                if vcl:
                    mu_current, logvar_current = model.get_variational_parameters()
                    kl_loss0 = compute_kl_divergence(mu_current, logvar_current, mu_previous0, logvar_previous0, reduction)/len_train_loader
                    loss -= task_id * kl_loss0.item()
                    for mu_previous, logvar_previous in zip(mu_previous_list, logvar_previous_list):
                        kl_loss = compute_kl_divergence(mu_current, logvar_current, mu_previous, logvar_previous, reduction)/len_train_loader
                        loss += kl_loss
                        
                loss.backward()
                optimizer.step()
                    
                running_loss += loss.item()
            
            overall_accuracy, task_accuracy = evaluate_model(model, test_datasets[:task_id+1], nb_simul, device, batch_size=batch_size)
            # print(overall_accuracy, task_accuracy)

            epoch_loss = running_loss / len(train_loader)
            task_losses.append(epoch_loss)
        if vcl:
            mu_previous, logvar_previous = mu_current.detach(), logvar_current.detach()

        overall_accuracy, task_accuracy = evaluate_model(model, test_datasets[:task_id+1], nb_simul, device, batch_size=batch_size)
        task_accuracies[task_id].extend(task_accuracy)
        losses.append(task_losses)
        overall_accuracies.append(overall_accuracy)

    return overall_accuracies, task_accuracies, losses, overall_baseline

def evaluate_model(model, test_datasets, nb_simul, device, batch_size=64):
    model.eval()
    all_preds = []
    all_labels = []
    task_accuracy = []

    for task_id, test_dataset in enumerate(test_datasets):
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        correct = 0
        total = 0

        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                for _ in range(nb_simul): # Averages on several runs
                    outputs = model(images)
                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
        accuracy = correct / total
        task_accuracy.append(accuracy)
        
    print([f"Task {i+1}: {task_accuracy[i]:.2f}" for i in range(len(task_accuracy))])
    overall_accuracy = sum(task_accuracy) / len(task_accuracy)
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    return overall_accuracy, task_accuracy

def save_plots_to_pdf(overall_accuracies, task_accuracies, losses, overall_baseline, filename, compute_baseline):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    for i in range(len(overall_accuracies)):
        task_accuracy = [task_accuracies[j][i] for j in range(i, len(task_accuracies))]
        plt.plot(range(i+1, len(task_accuracy)+i+1), task_accuracy, label=f'Task {i+1}', marker='o')
    plt.plot(range(1, len(overall_accuracies)+1), overall_accuracies, label='Overall Accuracy', linestyle='--', marker='o')
    if compute_baseline:
        plt.plot(range(1, len(overall_accuracies)+1), [overall_baseline]*len(overall_accuracies), label='Baseline', linestyle=':', marker='o')
    plt.xlabel('Tasks')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of Each Task After Learning')
    plt.legend()

    plt.subplot(1, 2, 2)
    for i, task_loss in enumerate(losses):
        plt.plot(task_loss, label=f'Task {i+1}', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss of Each Task During Learning Phase')

    plt.tight_layout()

    plt.savefig(os.path.join('outputs/PermutedMNIST/plots', filename))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Continual Learning with Deterministic Deep Networks')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs per task')
    parser.add_argument('--nb_simul', type=int, default=5, help='Number of simulations')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=100, help='Hidden size of the Bayesian model')
    parser.add_argument('--beta', type=float, default=1, help='Beta parameter for VCL')
    parser.add_argument('--vcl', action='store_true', help='Use Variational Continual Learning')
    parser.add_argument('--reduction', type=str, default='mean', help='Reduction method for the loss')
    parser.add_argument('--num_tasks', type=int, default=10, help='Number of tasks')
    parser.add_argument('--method', type=str, default='reparametrization', help='Method of forwarding, local parametrization or not')
    parser.add_argument('--coreset_size', type=int, default=0, help='Size of the coreset')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of data to use for training')
    parser.add_argument('--filename', type=str, default='', help='Name of the output PDF file')
    parser.add_argument('--bias', action='store_true', help='Use bias')
    parser.add_argument('--recompute_data', action='store_true', help='Recompute the Permuted MNIST data')
    parser.add_argument('--compute_baseline', action='store_true', help='Compute the baseline accuracy')
    parser.add_argument('--reuse_first', action='store_true', help='Reuse the first task model from files')
    parser.add_argument('--use_seed', action='store_true', help='Use a fixed seed for the random generator')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    

    tasks = [(i*10, (i+1)*10-1) for i in range(args.num_tasks)]
    
    if args.bias:
        print("Hmm Avoid bias...")
    if args.use_seed:
        print(f"Hmm Using seed is going to make multiple simulations unuseful... (I've not yet implemented subseeds for the simuls)")
        _ = set_seed(args.seed)
    
    if os.path.exists(f'./data/PermutedMNIST/data_{args.num_tasks}.pkl') and not args.recompute_data:
        print('Loading data from files. Use --recompute_data to force recompute.')
        with open(f'./data/PermutedMNIST/data_{args.num_tasks}.pkl', 'rb') as f:
            train_datasets, test_datasets = pickle.load(f)
    else:
        train_datasets, test_datasets = load_permuted_mnist(root='./data', train=True, num_tasks=args.num_tasks, train_ratio=args.train_ratio)
        # check if the data folder exists
        if not os.path.exists('./data/PermutedMNIST'):
            os.makedirs('./data/PermutedMNIST')
        with open(f'./data/PermutedMNIST/data_{args.num_tasks}.pkl', 'wb') as f:
            pickle.dump((train_datasets, test_datasets), f)
    
    if not os.path.exists('./outputs/PermutedMNIST/logs'):
        os.makedirs('./outputs/PermutedMNIST/logs')
    if not os.path.exists('./outputs/PermutedMNIST/models'):
        os.makedirs('./outputs/PermutedMNIST/models')
    if not os.path.exists('./outputs/PermutedMNIST/plots'):
        os.makedirs('./outputs/PermutedMNIST/plots')
    
    coreset_datasets, train_datasets = create_coresets(train_datasets, args.coreset_size)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
    
    # beta_scheduler = [args.beta*(1-0.9**i) for i in range(args.epochs)]
    # beta_scheduler = [0.01]*5 + [1]*5
    beta_scheduler = None

    model_bayesian = BayesianNN(28*28, args.hidden_size, 10, args.method, args.bias)
    model_bayesian2 = BayesianNN(28*28, args.hidden_size, 10, args.method, args.bias)
    overall_accuracies_bayesian, task_accuracies_bayesian, losses_bayesian, overall_baseline = continual_learning(model_bayesian, model_bayesian2, train_datasets, coreset_datasets, test_datasets, tasks, epochs_per_task=args.epochs, batch_size=args.batch_size, lr=args.lr, nb_simul=args.nb_simul, vcl=args.vcl, reduction=args.reduction, beta=args.beta, beta_scheduler=beta_scheduler, compute_baseline=args.compute_baseline, reuse_first=args.reuse_first, name=args.filename if args.filename else f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    full_file = args.filename+datetime.now().strftime('%Y%m%d_%H%M%S') if args.filename else f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_Continual_Bayesian_MNIST"
    with open(f'./outputs/PermutedMNIST/logs/{full_file}_{args.num_tasks}_{args.beta}_{args.batch_size}_{args.epochs}.pkl', 'wb') as f:
        pickle.dump((overall_accuracies_bayesian, task_accuracies_bayesian, losses_bayesian), f)
    
    save_plots_to_pdf(overall_accuracies_bayesian, task_accuracies_bayesian, losses_bayesian, overall_baseline, args.filename+datetime.now().strftime('%Y%m%d_%H%M%S')+".pdf" if args.filename else f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_Continual_Bayesian_MNIST.pdf", args.compute_baseline)
