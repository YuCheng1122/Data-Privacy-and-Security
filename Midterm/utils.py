import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
import os
import numpy as np
from rdp_accountant import compute_rdp, get_privacy_spent

def process_grad_batch(params, clipping=1):
    """
    Apply per-sample gradient clipping to ensure that each sample's gradient has a bounded L2 norm.
    
    This function computes the L2 norm of each per-sample gradient (stored in p.grad_batch)
    across all given parameters. It then scales down any sample whose norm exceeds the specified
    clipping threshold, averages the scaled gradients, and resets the per-sample gradients.
    
    Args:
        params (list): A list of model parameters, each having a 'grad_batch' attribute
                       containing per-sample gradients.
        clipping (float): The maximum allowed L2 norm for each sample's gradient.
    """
    # Determine batch size from the first parameter's per-sample gradient.
    n = params[0].grad_batch.shape[0]
    grad_norm_list = torch.zeros(n).cuda()
    
    # Compute the L2 norm for each sample's gradient across all parameters.
    for p in params:
        # Flatten the gradient of each sample.
        flat_g = p.grad_batch.reshape(n, -1)
        current_norm_list = torch.norm(flat_g, dim=1)
        grad_norm_list += torch.square(current_norm_list)
    grad_norm_list = torch.sqrt(grad_norm_list)
    
    # Compute scaling factors; if a sample's norm exceeds the clipping threshold, scale it down.
    scaling = clipping / grad_norm_list
    scaling[scaling > 1] = 1  # Do not scale up if the norm is below the threshold.
    
    # Apply scaling to each parameter's per-sample gradient and aggregate them.
    for p in params:
        # Reshape scaling to be broadcastable for p.grad_batch.
        p_dim = len(p.shape)
        scaling_reshaped = scaling.view([n] + [1] * p_dim)
        p.grad_batch *= scaling_reshaped
        # Aggregate per-sample gradients by computing the mean.
        p.grad = torch.mean(p.grad_batch, dim=0)
        # Reset the per-sample gradient storage for the next iteration.
        p.grad_batch.mul_(0.)

def get_data_loader(dataset, batchsize):
    """
    Create data loaders for different datasets based on the provided dataset name.
    
    For 'svhn', the training set is split into 'train' and 'extra' to allow full batch loading for concatenation;
    for 'mnist', the standard training and test splits are used;
    the default case handles CIFAR10 with standard training and test transformations.
    
    Args:
        dataset (str): Name of the dataset ('svhn', 'mnist', or others defaulting to CIFAR10).
        batchsize (int): Batch size for the training/test loaders.
    
    Returns:
        If dataset is 'svhn': returns (trainloader, extraloader, testloader, total_train_size, test_size).
        Otherwise: returns (trainloader, testloader, trainset_size, testset_size).
    """
    if dataset == 'svhn':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        # Load the 'train' split as one set.
        trainset = torchvision.datasets.SVHN('./data', split='train', download=True, transform=transform)
        # Load the complete train set in one large batch.
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=73257, shuffle=True, num_workers=0)
        
        # Load the 'extra' split as additional training data.
        extraset = torchvision.datasets.SVHN('./data', split='extra', download=True, transform=transform)
        extraloader = torch.utils.data.DataLoader(
            extraset, batch_size=531131, shuffle=True, num_workers=0)
        
        # Load the test set.
        testset = torchvision.datasets.SVHN('./data', split='test', download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batchsize, shuffle=False, num_workers=0)
        return trainloader, extraloader, testloader, len(trainset) + len(extraset), len(testset)
    
    elif dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batchsize, shuffle=True, num_workers=2)
        
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batchsize, shuffle=False, num_workers=2)
        return trainloader, testloader, len(trainset), len(testset)
    
    else:
        # Default: Use CIFAR10 dataset.
        transform_train = transforms.Compose([
            # Optional augmentations (RandomCrop, RandomHorizontalFlip) may be added here.
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batchsize, shuffle=True, num_workers=2)
        
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batchsize, shuffle=False, num_workers=2)
        return trainloader, testloader, len(trainset), len(testset)

def loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rdp_orders=32, rgp=True):
    """
    Iteratively adjust the noise standard deviation (sigma) to achieve a target privacy budget.
    
    This function uses Renyi Differential Privacy (RDP) accounting to compute the privacy loss (epsilon)
    for a given sigma, and iteratively decreases or increases sigma by the specified interval until
    the computed epsilon meets the target constraint.
    
    Args:
        q (float): Sampling probability (batch size divided by dataset size).
        T (int): Total number of training steps.
        eps (float): Target privacy budget (epsilon).
        delta (float): Target delta for differential privacy.
        cur_sigma (float): Current estimate of the noise standard deviation.
        interval (float): Adjustment interval to update sigma.
        rdp_orders (float, optional): The maximum order to consider in RDP computation. Default is 32.
        rgp (bool, optional): Flag indicating if residual gradients are used (affects sensitivity). Default is True.
    
    Returns:
        tuple: (cur_sigma, previous_eps) where cur_sigma is the updated noise scale
               and previous_eps is the privacy loss at the last acceptable sigma.
    """
    while True:
        orders = np.arange(2, rdp_orders, 0.1)
        steps = T
        # Compute RDP; when using residual gradients, sensitivity is scaled by sqrt(2).
        if rgp:
            rdp = compute_rdp(q, cur_sigma, steps, orders) * 2
        else:
            rdp = compute_rdp(q, cur_sigma, steps, orders)
        # Compute the overall privacy loss epsilon from the RDP values.
        cur_eps, _, opt_order = get_privacy_spent(orders, rdp, target_delta=delta)
        # If the current epsilon is below target and sigma can be further decreased, reduce sigma.
        if cur_eps < eps and cur_sigma > interval:
            cur_sigma -= interval
            previous_eps = cur_eps
        else:
            # Otherwise, increase sigma slightly and exit the loop.
            cur_sigma += interval
            break
    return cur_sigma, previous_eps

def get_sigma(q, T, eps, delta, init_sigma=10, interval=1., rgp=True):
    """
    Determine an appropriate noise standard deviation (sigma) to satisfy the differential privacy requirement.
    
    The function repeatedly refines the estimate of sigma using decreasing intervals for higher precision.
    
    Args:
        q (float): Sampling probability (batch size / dataset size).
        T (int): Total number of training steps.
        eps (float): Target privacy budget (epsilon).
        delta (float): Delta parameter for differential privacy.
        init_sigma (float, optional): Initial sigma guess. Default is 10.
        interval (float, optional): Initial adjustment interval. Default is 1.0.
        rgp (bool, optional): Indicates whether residual gradients are used (affects sensitivity). Default is True.
    
    Returns:
        tuple: (cur_sigma, previous_eps) where cur_sigma is the determined noise scale and previous_eps is 
               the corresponding privacy loss.
    """
    cur_sigma = init_sigma
    # Coarse adjustment.
    cur_sigma, _ = loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rgp=rgp)
    interval /= 10  # Finer adjustment.
    cur_sigma, _ = loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rgp=rgp)
    interval /= 10  # Further fine-tuning.
    cur_sigma, previous_eps = loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rgp=rgp)
    return cur_sigma, previous_eps

def differentially_private_pca(data, n_components=60, epsilon=1.0, delta=1e-5):
    """
    Perform PCA in a manner that satisfies differential privacy.
    
    This implementation first flattens the input data and normalizes each sample to have unit norm.
    It then computes the covariance matrix of the normalized data, adds calibrated Gaussian noise to ensure
    differential privacy, and finally performs eigen-decomposition to obtain the principal components.
    
    Args:
        data (torch.Tensor): Input data tensor (e.g., images) to perform PCA on.
        n_components (int, optional): The number of principal components to retain. Default is 60.
        epsilon (float, optional): The privacy budget for the PCA computation. Default is 1.0.
        delta (float, optional): The delta parameter for differential privacy. Default is 1e-5.
    
    Returns:
        torch.FloatTensor: A tensor of shape (original_dim, n_components) containing the principal components.
    """
    # Convert data to a NumPy array and flatten each sample.
    data_np = data.cpu().numpy().reshape(len(data), -1)
    
    # Normalize data so that each sample is a unit vector.
    norms = np.linalg.norm(data_np, axis=1, keepdims=True)
    normalized_data = data_np / norms
    
    # Compute the covariance matrix of the normalized data.
    cov_matrix = np.dot(normalized_data.T, normalized_data) / len(normalized_data)
    
    # Define sensitivity for unit vectors.
    sensitivity = 2
    # Calculate the noise scale (sigma) based on the Gaussian mechanism.
    sigma = np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    # Sample Gaussian noise and add it to the covariance matrix.
    noise = np.random.normal(0, sigma * sensitivity, cov_matrix.shape)
    # Ensure the added noise is symmetric.
    noise = (noise + noise.T) / 2
    
    # Obtain the noisy covariance matrix.
    noisy_cov = cov_matrix + noise
    
    # Perform eigen-decomposition on the noisy covariance matrix.
    eigenvalues, eigenvectors = np.linalg.eigh(noisy_cov)
    
    # Sort eigenvectors in descending order by their corresponding eigenvalues.
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select the top 'n_components' eigenvectors.
    components = eigenvectors[:, :n_components]
    
    # Convert the principal components to a Torch FloatTensor.
    return torch.FloatTensor(components)

def checkpoint(net, acc, epoch, sess):
    """
    Save the current state of the model along with training metadata to a checkpoint file.
    
    The checkpoint includes the model parameters, current accuracy, epoch number, and the RNG state.
    
    Args:
        net (nn.Module): The neural network model.
        acc (float): The accuracy achieved so far.
        epoch (int): The current training epoch.
        sess (str): A session identifier for naming the checkpoint file.
    """
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state(),
    }
    
    # Create a checkpoint directory if it doesn't exist.
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/' + sess + '.ckpt')

def adjust_learning_rate(optimizer, init_lr, epoch, all_epoch):
    """
    Adjust the learning rate according to the training progress.
    
    The learning rate is decreased at specific milestones during training. For example,
    it remains constant for the first half of the training, is reduced by a factor of 10
    for the next quarter, and finally by a factor of 100 for the remaining training epochs.
    
    Args:
        optimizer (torch.optim.Optimizer): The optimizer whose learning rate will be adjusted.
        init_lr (float): The initial learning rate.
        epoch (int): The current epoch.
        all_epoch (int): The total number of training epochs.
    
    Returns:
        float: The new learning rate after adjustment.
    """
    if epoch < all_epoch * 0.5:
        decay = 1.0
    elif epoch < all_epoch * 0.75:
        decay = 10.0
    else:
        decay = 100.0

    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr / decay
    return init_lr / decay
