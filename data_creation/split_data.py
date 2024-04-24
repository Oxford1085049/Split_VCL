from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def load_split_mnist(root='./data', train_ratio=0.8):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_dataset = datasets.MNIST(root=root, train=True, transform=transform, download=True)
    mnist_size = len(mnist_dataset)

    train_datasets = []
    test_datasets = []

    for task_id in tqdm(range(5), desc='Computing Split MNIST'):
        # Define the digits for the current task
        digit_1 = task_id * 2
        digit_2 = task_id * 2 + 1

        # Filter indices for the current digits
        digit_1_indices = [idx for idx, (_, label) in enumerate(mnist_dataset) if label == digit_1 or label == digit_2]

        # Split indices into train and test sets
        digit_1_train_indices, digit_1_test_indices = train_test_split(digit_1_indices, train_size=train_ratio, shuffle=True)

        # Create train and test subsets
        digit_1_train_subset = Subset(mnist_dataset, digit_1_train_indices)
        digit_1_test_subset = Subset(mnist_dataset, digit_1_test_indices)

        # Combine subsets for each task
        train_datasets.append(digit_1_train_subset)
        test_datasets.append(digit_1_test_subset)

    return train_datasets, test_datasets
