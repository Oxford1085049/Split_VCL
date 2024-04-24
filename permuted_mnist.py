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



def continual_learning(model, train_datasets, coreset_datasets, test_datasets, tasks, epochs_per_task=5, batch_size=64, lr=0.001, nb_simul=10, vcl=False, reduction='sum', beta=1, beta_scheduler=None, compute_baseline=False, reuse_first=False, name=None):
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
    model.train()
    overall_accuracies = []
    task_accuracies = []
    losses = []
    

    for task_id, (task_start, task_end) in enumerate(tasks):
        kl_losses = np.array([0.]*epochs_per_task)
        log_losses = np.array([0.]*epochs_per_task)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        if reuse_first and task_id == 0:
            model.load_state_dict(torch.load(f'./outputs/PermutedMNIST/models/model_{0}.pt'))
            mu_current, logvar_current = model.get_variational_parameters()
            mu_previous, logvar_previous = mu_current.detach(), logvar_current.detach()
            continue
        # model.reset_parameters() # should i let it???
        concatenated_train_dataset = ConcatDataset([train_datasets[task_id]] + coreset_datasets[:task_id]) # coreset not yet fully implemented, evaluation missing
        train_loader = DataLoader(concatenated_train_dataset, batch_size=batch_size, shuffle=True)
        len_train_loader = len(train_loader)
        # print(len_train_loader, len(train_datasets[task_id]))
        task_accuracies.append([])
        task_losses = []
        for epoch in tqdm(range(epochs_per_task), desc=f'Task {task_id+1}'):
            running_loss = 0.0
            owief = 0
            for images, labels in train_loader:
                owief += 1
                images, labels = images.to(device), labels.to(device)
                for _ in range(nb_simul):
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = torch.tensor(0.0)
                    log_likelihood_loss = -torch.log_softmax(outputs, dim=1)[range(labels.size(0)), labels].mean()
                    loss += log_likelihood_loss
                    log_losses[epoch] += log_likelihood_loss.item()
                    if vcl:
                        mu_current, logvar_current = model.get_variational_parameters()
                        if task_id >= 0: # not needed, was just to test without the first kl div with prior
                            kl_loss = beta_scheduler[epoch] * compute_kl_divergence(mu_current, logvar_current, mu_previous, logvar_previous, reduction)/len_train_loader
                            # print(kl_loss.item())
                            if kl_loss < 0: # too small, approximations error
                                continue
                                raise ValueError(f"KL Loss is negative: {kl_loss}")
                            loss += kl_loss
                            kl_losses[epoch] += kl_loss.item()
                            
                    loss.backward()
                    optimizer.step()
                    
                    ##### GRADIENT CLIPPING ##### (doesn't improve the results)
                    # log_likelihood_loss = -torch.log_softmax(outputs, dim=1)[range(labels.size(0)), labels].mean()
                    # if task_id == 0:
                    #     log_likelihood_loss.backward(retain_graph=True)
                    #     log_gradients = [param.grad.clone().detach() for param in model.parameters()]
                    # loss += log_likelihood_loss
                    # local_losses.append(loss.item())
                    # if vcl:
                    #     mu_current, logvar_current = model.get_variational_parameters()
                    #     if task_id > 0:
                    #         kl_loss = beta * compute_kl_divergence(mu_current, logvar_current, mu_previous, logvar_previous, reduction)#/(len(train_datasets[task_id-1]))
                    #         local_kl_losses.append(kl_loss.item())
                    #         kl_loss.backward(retain_graph=True)
                    #         max_norm = 1000
                    #         kl_gradients = []
                    #         for param in model.parameters():
                    #             kl_gradients.append(param.grad.clone())
                    #         # Gradient clipping!!!
                    #         total_norm = 0.0
                    #         for grad in kl_gradients:
                    #             total_norm += grad.norm().item() ** 2
                    #         total_norm = total_norm ** 0.5
                    #         clip_coef = max_norm / (total_norm + 1e-6)
                    #         if clip_coef < 1:
                    #             print("Hmm")
                    #             for grad in kl_gradients:
                    #                 grad.mul_(clip_coef)
                                    
                    #         index = 0
                    #         for param in model.parameters():
                    #             param.grad = kl_gradients[index]
                    #             index += 1
                    #         optimizer.step()
                    #         # if task_id == 1:
                    #         #     print(kl_loss.item(), loss.item())
                    #         if kl_loss < 0:
                    #             continue
                    #             raise ValueError(f"KL Loss is negative: {kl_loss}")
                    #         loss += kl_loss
                    #         # print(f"KL Los {owief}: {kl_loss.item()}")
                    # index = 0
                    # for param in model.parameters():
                    #     param.grad = log_gradients[index]  # Restore gradients before KL divergence gradient
                    #     index += 1
                    # # optimizer.step()
                    ##### END OF GRADIENT CLIPPING #####
                
                    
                running_loss += loss.item()
            
            overall_accuracy, task_accuracy = evaluate_model(model, test_datasets[:task_id+1], nb_simul, device, batch_size=batch_size)
            # print(overall_accuracy, task_accuracy)

            epoch_loss = running_loss / len(train_loader)
            task_losses.append(epoch_loss)
        if vcl:
            mu_previous, logvar_previous = mu_current.detach(), logvar_current.detach()
        # save the model
        torch.save(model.state_dict(), f'./outputs/PermutedMNIST/models/model_{task_id}.pt')
        # save kl_losses and log_losses
        with open(f'./outputs/PermutedMNIST/logs/losses_{beta}_{task_id}_{epoch}_{batch_size}_{name}.pkl', 'wb') as f:
            pickle.dump((kl_losses, log_losses), f)
        # plot them on same plot
        # print(kl_losses, log_losses)
        plt.plot(range(1, epochs_per_task+1), kl_losses, label='KL Loss')
        plt.plot(range(1, epochs_per_task+1), log_losses, label='Log Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('KL and Log Losses')
        plt.legend()
        plt.savefig(f'./outputs/PermutedMNIST/plots/losses_{beta}_{task_id}_{epoch}_{batch_size}_{name}.pdf')
        plt.close()

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
        print(f"Hmm Using seed is going to make multiple simulations unuseful... (I've not yet implemented subseeds for the simuls) Make sure the batch size is high enough to compensate")
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
    overall_accuracies_bayesian, task_accuracies_bayesian, losses_bayesian, overall_baseline = continual_learning(model_bayesian, train_datasets, coreset_datasets, test_datasets, tasks, epochs_per_task=args.epochs, batch_size=args.batch_size, lr=args.lr, nb_simul=args.nb_simul, vcl=args.vcl, reduction=args.reduction, beta=args.beta, beta_scheduler=beta_scheduler, compute_baseline=args.compute_baseline, reuse_first=args.reuse_first, name=args.filename if args.filename else f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    full_file = args.filename+datetime.now().strftime('%Y%m%d_%H%M%S') if args.filename else f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_Continual_Bayesian_MNIST"
    with open(f'./outputs/PermutedMNIST/logs/{full_file}_{args.num_tasks}_{args.beta}_{args.batch_size}_{args.epochs}.pkl', 'wb') as f:
        pickle.dump((overall_accuracies_bayesian, task_accuracies_bayesian, losses_bayesian), f)
    
    save_plots_to_pdf(overall_accuracies_bayesian, task_accuracies_bayesian, losses_bayesian, overall_baseline, args.filename+datetime.now().strftime('%Y%m%d_%H%M%S')+".pdf" if args.filename else f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_Continual_Bayesian_MNIST.pdf", args.compute_baseline)
