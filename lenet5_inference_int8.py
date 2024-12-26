# -*- coding: utf-8 -*-
"""
@author: Phuoc Huu
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import torch.quantization
from torch.quantization import QuantStub, DeQuantStub
import time
import os

# Model gốc để train
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.quant = QuantStub()
        self.conv1 = nn.Conv2d(1, 6, 1, 1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        self.relu2 = nn.ReLU()
        self.pooling = nn.AvgPool2d(2,2)
        self.fc1 = nn.Linear(400, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pooling(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        x = self.dequant(x)
        output = F.log_softmax(x, dim=1)
        return output

# Model INT8 để test
class LeNet5_INT8(nn.Module):
    def __init__(self):
        super(LeNet5_INT8, self).__init__()
        self.conv1_weight = None
        self.conv1_bias = None
        self.conv2_weight = None 
        self.conv2_bias = None
        self.fc1_weight = None
        self.fc1_bias = None
        self.fc2_weight = None
        self.fc2_bias = None
        self.fc3_weight = None
        self.fc3_bias = None
        
        # Load weights và scales
        self.load_quantized_params()
        
    def load_quantized_params(self):
        weights_dir = "data/weights_int8/"
        
        def load_params(name):
            weights = np.loadtxt(f"{weights_dir}w_{name}.txt", dtype=np.int8)
            bias = np.loadtxt(f"{weights_dir}b_{name}.txt", dtype=np.int8)
            w_scale = np.loadtxt(f"{weights_dir}w_{name}_scale.txt")
            b_scale = np.loadtxt(f"{weights_dir}b_{name}_scale.txt")
            return weights, bias, w_scale, b_scale

        # Load tất cả params
        c1_w, c1_b, c1_ws, c1_bs = load_params("conv1")
        c2_w, c2_b, c2_ws, c2_bs = load_params("conv2") 
        f1_w, f1_b, f1_ws, f1_bs = load_params("fc1")
        f2_w, f2_b, f2_ws, f2_bs = load_params("fc2")
        f3_w, f3_b, f3_ws, f3_bs = load_params("fc3")

        # Reshape weights về đúng kích thước
        self.conv1_weight = torch.from_numpy(c1_w).reshape(6, 1, 1, 1)
        self.conv1_bias = torch.from_numpy(c1_b)
        self.conv2_weight = torch.from_numpy(c2_w).reshape(16, 6, 5, 5) 
        self.conv2_bias = torch.from_numpy(c2_b)
        self.fc1_weight = torch.from_numpy(f1_w).reshape(120, 400)
        self.fc1_bias = torch.from_numpy(f1_b)
        self.fc2_weight = torch.from_numpy(f2_w).reshape(84, 120)
        self.fc2_bias = torch.from_numpy(f2_b)
        self.fc3_weight = torch.from_numpy(f3_w).reshape(10, 84)
        self.fc3_bias = torch.from_numpy(f3_b)

        # Lưu scale factors
        self.scales = {
            'conv1': (c1_ws, c1_bs),
            'conv2': (c2_ws, c2_bs),
            'fc1': (f1_ws, f1_bs),
            'fc2': (f2_ws, f2_bs),
            'fc3': (f3_ws, f3_bs)
        }

    def forward(self, x):
        # Normalize input như code C
        x = (x - 0.1307) / 0.3081
        
        # Conv1
        x = F.conv2d(x, self.conv1_weight.float(), self.conv1_bias.float())
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)
        
        # Conv2 
        x = F.conv2d(x, self.conv2_weight.float(), self.conv2_bias.float())
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # FC1
        x = F.linear(x, self.fc1_weight.float(), self.fc1_bias.float())
        x = F.relu(x)
        
        # FC2
        x = F.linear(x, self.fc2_weight.float(), self.fc2_bias.float())
        x = F.relu(x)
        
        # FC3
        x = F.linear(x, self.fc3_weight.float(), self.fc3_bias.float())
        
        return F.log_softmax(x, dim=1)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def test_int8(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nINT8 Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def quantize_and_save_weights(model, test_loader):
    weights_dir = "data/weights_int8/"
    os.makedirs(weights_dir, exist_ok=True)

    model.eval()

    def save_quantized_tensor(tensor, filename):
        tensor_np = tensor.detach().cpu().numpy()
        max_val = np.abs(tensor_np).max()
        scale = 127.0 / max(max_val, 1e-8)
        tensor_int8 = np.clip(np.round(tensor_np * scale), -128, 127).astype(np.int8)
        np.savetxt(f"{weights_dir}{filename}", tensor_int8.reshape(-1), fmt='%d')
        np.savetxt(f"{weights_dir}{filename}_scale.txt", [scale], fmt='%.10f')

    state_dict = model.state_dict()
    
    # Save conv1
    save_quantized_tensor(state_dict['conv1.weight'], 'w_conv1.txt')
    save_quantized_tensor(state_dict['conv1.bias'], 'b_conv1.txt')
    
    # Save conv2
    save_quantized_tensor(state_dict['conv2.weight'], 'w_conv2.txt')
    save_quantized_tensor(state_dict['conv2.bias'], 'b_conv2.txt')
    
    # Save fc1
    save_quantized_tensor(state_dict['fc1.weight'], 'w_fc1.txt')
    save_quantized_tensor(state_dict['fc1.bias'], 'b_fc1.txt')
    
    # Save fc2
    save_quantized_tensor(state_dict['fc2.weight'], 'w_fc2.txt')
    save_quantized_tensor(state_dict['fc2.bias'], 'b_fc2.txt')
    
    # Save fc3
    save_quantized_tensor(state_dict['fc3.weight'], 'w_fc3.txt')
    save_quantized_tensor(state_dict['fc3.bias'], 'b_fc3.txt')

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                      'pin_memory': True,
                      'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = LeNet5().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    # Quantize và lưu weights
    quantize_and_save_weights(model, test_loader)

    # Test với model INT8
    model_int8 = LeNet5_INT8().to(device)
    test_int8(model_int8, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_lenet5.pt")

if __name__ == '__main__':
    main()
