import torch
from torchvision import datasets, transforms
from tqdm import tqdm

def load_permuted_mnist(root='./data', train=True, num_tasks=10, train_ratio=0.8):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    mnist_dataset = datasets.MNIST(root=root, train=train, transform=transform, download=True)
    mnist_size = len(mnist_dataset)

    train_datasets = []
    test_datasets = []

    for task_id in tqdm(range(num_tasks), desc='Computing Permuted MNIST'):
        perm = torch.randperm(28 * 28)
        train_size = int(train_ratio * mnist_size)
        train_data = []
        test_data = []

        for i in range(train_size):
            img, label = mnist_dataset[i]
            img = img.view(-1)[perm].view(1, 28, 28)
            train_data.append((img, label))

        for i in range(train_size, mnist_size):
            img, label = mnist_dataset[i]
            img = img.view(-1)[perm].view(1, 28, 28)
            test_data.append((img, label))

        train_datasets.append(train_data)
        test_datasets.append(test_data)

    return train_datasets, test_datasets