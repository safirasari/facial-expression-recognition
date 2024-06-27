from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms

classes = ('neutral', 'focused', 'angry', 'happy')

# Transformation to ensure consistency
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),    # 1 channel for grayscale
    transforms.ToTensor(),                          # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))            # Center data around 0 (instead of [0,1])
])

# Split dataset into sets of train, validation, and test
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

dataset = ImageFolder(root='./datasets', transform=transform)
middleaged_dataset = ImageFolder(root='./datasets', transform=transform)
senior_dataset = ImageFolder(root='./datasets', transform=transform)
young_dataset = ImageFolder(root='./datasets', transform=transform)
female_dataset = ImageFolder(root='./datasets', transform=transform)
male_dataset = ImageFolder(root='./datasets', transform=transform)
other_dataset = ImageFolder(root='./datasets', transform=transform)

num_samples = len(dataset)
num_train = int(train_ratio * num_samples)
num_val = int(val_ratio * num_samples)
num_test = num_samples - num_train - num_val

# Create indices for each split
indices = list(range(num_samples))
train_indices, remaining_indices = train_test_split(indices, test_size=(val_ratio + test_ratio), random_state=42)
val_indices, test_indices = train_test_split(remaining_indices, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)

# Create data samplers
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

# Define batch sizes
train_batch_size = 32
val_batch_size = 32
test_batch_size = 1000

# Create DataLoaders for each set
train_loader = DataLoader(dataset, batch_size=train_batch_size, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=val_batch_size, sampler=val_sampler)
test_loader = DataLoader(dataset, batch_size=test_batch_size, sampler=test_sampler)
