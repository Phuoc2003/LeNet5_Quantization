# # -*- coding: utf-8 -*-
# """
# Created on Fri Jan  5 10:28:48 2024

# @author: Phat_Dang
# """

# import argparse
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.optim.lr_scheduler import StepLR
# import numpy as np
# import time
# import os
# from pathlib import Path
# #import adamod
# class LeNet5(nn.Module):
#     def __init__(self):
#         super(LeNet5, self).__init__()
#         self.conv1 = nn.Conv2d(1, 6, 1, 1)
#         self.conv2 = nn.Conv2d(6, 16, 5, 1)
#         self.pooling = nn.AvgPool2d(2,2)
#         self.fc1 = nn.Linear(400, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.pooling(x)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = self.pooling(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         x = F.relu(x)
#         x = self.fc3(x)
#         output = torch.log_softmax(x, dim=1)
#         return output
    
# class LeNet5_quantize(nn.Module):
#     def __init__(self):
#         super(LeNet5_quantize, self).__init__()
#         self.quant = torch.quantization.QuantStub()
#         self.conv1 = nn.Conv2d(1, 6, 1, 1)
#         self.conv2 = nn.Conv2d(6, 16, 5, 1)
#         self.pooling = nn.AvgPool2d(2,2)
#         self.fc1 = nn.Linear(400, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#         self.dequant = torch.quantization.DeQuantStub()

#     def forward(self, x):
#         x = self.quant(x)
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.pooling(x)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = self.pooling(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         x = F.relu(x)
#         x = self.fc3(x)
#         x = self.dequant(x)
#         output = torch.log_softmax(x, dim=1)
#         return output

# def train(args, model, device, train_loader, optimizer, epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % args.log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))
#             if args.dry_run:
#                 break


# def test(model, device, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
#             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()
#     test_loss /= len(test_loader.dataset)

#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))

# def print_size_of_model(model):
#     torch.save(model.state_dict(), "temp_delme.p")
#     print('Size (KB):', os.path.getsize("temp_delme.p")/1e3)
#     os.remove('temp_delme.p')
    

# def save_weights_to_txt(model):
#     weights_dir = "data/weights_int8/"
#     os.makedirs(weights_dir, exist_ok=True)

#     state_dict = model.state_dict()
    
#     def process_weight(tensor):
#         if hasattr(tensor, 'int_repr'):
#             return tensor.int_repr().cpu().detach().numpy()
#         else:
#             return tensor.cpu().detach().numpy()

#     def quantize_bias(tensor):
#         # Quantize về int32
#         return tensor.detach().cpu().numpy()

#     # Conv1
#     w_conv1 = process_weight(state_dict['conv1.weight'])
#     b_conv1 = state_dict['conv1.bias']

    
#     b_conv1_int32 = quantize_bias(b_conv1)
    
#     np.savetxt(weights_dir + "w_conv1.txt", w_conv1.reshape(-1), fmt='%d')
#     np.savetxt(weights_dir + "b_conv1.txt", b_conv1_int32, fmt='%f')

#     # Conv2
#     w_conv2 = process_weight(state_dict['conv2.weight'])
#     b_conv2 = state_dict['conv2.bias']
    
#     b_conv2_int32 = quantize_bias(b_conv2)
    
#     np.savetxt(weights_dir + "w_conv2.txt", w_conv2.reshape(-1), fmt='%d')
#     np.savetxt(weights_dir + "b_conv2.txt", b_conv2_int32, fmt='%f')

#     # FC layers
#     for fc_name in ['fc1', 'fc2', 'fc3']:
#         packed_params = state_dict[f'{fc_name}._packed_params._packed_params']
#         weight = process_weight(packed_params[0])
#         bias = packed_params[1]
        
#         b_int32 = quantize_bias(bias)
        
#         np.savetxt(f"{weights_dir}w_{fc_name}.txt", weight.reshape(-1), fmt='%d')
#         np.savetxt(f"{weights_dir}b_{fc_name}.txt", b_int32, fmt='%f')

#     print("Weights and biases saved to", weights_dir)




# def main():
#     # Training settings
#     parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
#     parser.add_argument('--batch-size', type=int, default=32, metavar='N',
#                         help='input batch size for training (default: 64)')
#     parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
#                         help='input batch size for testing (default: 1000)')
#     parser.add_argument('--epochs', type=int, default=1, metavar='N',
#                         help='number of epochs to train (default: 1)')
#     parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
#                         help='learning rate (default: 1.0)')
#     parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
#                         help='Learning rate step gamma (default: 0.7)')
#     parser.add_argument('--no-cuda', action='store_true', default=False,
#                         help='disables CUDA training')
#     parser.add_argument('--dry-run', action='store_true', default=False,
#                         help='quickly check a single pass')
#     parser.add_argument('--seed', type=int, default=1, metavar='S',
#                         help='random seed (default: 1)')
#     parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                         help='how many batches to wait before logging training status')
#     parser.add_argument('--save-model', action='store_true', default=True,
#                         help='For Saving the current Model')
#     args = parser.parse_args()
#     use_cuda = not args.no_cuda and torch.cuda.is_available()

#     torch.manual_seed(args.seed)

#     device = torch.device("cuda" if use_cuda else "cpu")
#     #device = torch.device("cpu")

#     train_kwargs = {'batch_size': args.batch_size}
#     test_kwargs = {'batch_size': args.test_batch_size}
#     if use_cuda:
#         cuda_kwargs = {'num_workers': 1,
#                        'pin_memory': True,
#                        'shuffle': True}
#         train_kwargs.update(cuda_kwargs)
#         test_kwargs.update(cuda_kwargs)

#     transform=transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,))
#         ])
#     dataset1 = datasets.MNIST('data', train=True, download=True,
#                        transform=transform)
#     dataset2 = datasets.MNIST('data', train=False,
#                        transform=transform)
#     train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
#     test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

#     model = LeNet5().to(device)
#     optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
#     scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
#     ### Training Phase
#     MODEL_FILENAME = "data/lenet5_model.pt"
#     if Path(MODEL_FILENAME).exists():
#         model.load_state_dict(torch.load(MODEL_FILENAME))
#         print('Loaded model from disk')
#     else: 
#         for epoch in range(1, args.epochs + 1):
#             train(args, model, device, train_loader, optimizer, epoch)
#             test(model, device, test_loader)
#             scheduler.step()
#             ### Save the model
#             if args.save_model:
#                 torch.save(model.state_dict(), MODEL_FILENAME)
    
#     ### Load the model from lenet5_model.pt file
#     print(model)
#     model.load_state_dict(torch.load("data/lenet5_model.pt"))
#     keys = model.state_dict().keys()
    
    
#     ### Export all model's weights to *.txt file   
#     for steps, weights in enumerate(keys):
#         w = model.state_dict()[weights].data
#         w_reshaped = w.reshape(w.shape[0],-1)
#         np.savetxt(r'data/weights/' + str(weights) + '.txt', w_reshaped, fmt='%f')
     
#     print('Weights before quantization')
#     print(model.conv1.weight)
#     print(model.conv1.weight.dtype)
    
#     print('Size of the model before quantization')
#     print_size_of_model(model)
    
#     ### Inference Phase   
#     print('Testing before quantize...')
#     test(model, device, test_loader)
    
#     model_quantized = LeNet5_quantize().to(device)
#     # Copy weights from unquantized model
#     model_quantized.load_state_dict(model.state_dict())
#     model_quantized.eval()

#     qconfig = torch.quantization.QConfig(
#         activation=torch.quantization.default_observer,
#         weight=torch.quantization.default_weight_observer
#     )
#     # Áp dụng qconfig cho model
#     model_quantized.qconfig = qconfig

#     # Chuẩn bị model cho quantization
#     torch.quantization.prepare(model_quantized, inplace=True) # Insert observers
#     print(model_quantized)
    
#     print('Calibrate...')
#     test(model_quantized, device, test_loader)
    
#     print(f'Check statistics of the various layers')
#     print(model_quantized)
    
#     #Quantize the model using the statistics collected
#     model_quantized = torch.ao.quantization.convert(model_quantized)

#     print(f'Check statistics of the various layers after quantized')
#     print(model_quantized)
    
#     print('Weights after quantization')
#     print(torch.int_repr(model_quantized.conv1.weight()))

#     print('Original weights: ')
#     print(model.conv1.weight)
#     print(model.conv1.bias)
#     print('')
#     print(f'Dequantized weights: ')
#     print(torch.dequantize(model_quantized.conv1.weight()))
#     print('')
    
#     print('Size of the model after quantization')
#     print_size_of_model(model_quantized)
    
#     MODEL_FILENAME = "data/lenet5_quantized_model.pt"
#     if args.save_model:
#         torch.save(model_quantized.state_dict(), MODEL_FILENAME)
        
#     print('Testing the model after quantization')
#     test(model_quantized, device, test_loader)
    
#     save_weights_to_txt(model_quantized)
        
# if __name__ == '__main__':
#    main()

# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 10:28:48 2024

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
from pathlib import Path
import time
#import adamod
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 1, 1)
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        self.pooling = nn.AvgPool2d(2,2)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pooling(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = torch.log_softmax(x, dim=1)
        return output

def quantize_tensor(tensor, num_bits=8):
    # Tìm giá trị max tuyệt đối
    max_abs = torch.max(torch.abs(tensor))
    
    # Tính scale factor
    scale = 0.1
    
    # Quantize
    quantized = torch.round(tensor / scale) * scale
    
    # Clip để đảm bảo nằm trong range của int8
    quantized = torch.clamp(quantized, -max_abs, max_abs)
    
    return quantized, scale

def save_weights_to_txt(model):
    weights_dir = "data/weights/"  # Directory to save weights
    weights_dir_int8 = "data/weights_int8/"  # Directory to save quantized weights
    import os
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(weights_dir_int8, exist_ok=True)

    # Dictionary để lưu scale factors
    scales = {}
    
    # Conv1 layer
    w_conv1, scale = quantize_tensor(model.state_dict()['conv1.weight'])
    scales['conv1'] = scale
    np.savetxt(weights_dir + "w_conv1.txt", w_conv1.cpu().numpy().reshape(-1), fmt='%f')
    np.savetxt(weights_dir_int8 + "w_conv1.txt", 
               (w_conv1/scale).cpu().numpy().astype(np.int8).reshape(-1), fmt='%d')
    np.savetxt(weights_dir + "b_conv1.txt", model.state_dict()['conv1.bias'].cpu().numpy(), fmt='%f')
    
    # Conv2 layer
    w_conv2, scale = quantize_tensor(model.state_dict()['conv2.weight'])
    scales['conv2'] = scale
    np.savetxt(weights_dir + "w_conv2.txt", w_conv2.cpu().numpy().reshape(-1), fmt='%f')
    np.savetxt(weights_dir_int8 + "w_conv2.txt", 
               (w_conv2/scale).cpu().numpy().astype(np.int8).reshape(-1), fmt='%d')
    np.savetxt(weights_dir + "b_conv2.txt", model.state_dict()['conv2.bias'].cpu().numpy(), fmt='%f')

    # FC1 layer
    w_fc1, scale = quantize_tensor(model.state_dict()['fc1.weight'])
    scales['fc1'] = scale
    np.savetxt(weights_dir + "w_fc1.txt", w_fc1.cpu().numpy().reshape(-1), fmt='%f')
    np.savetxt(weights_dir_int8 + "w_fc1.txt", 
               (w_fc1/scale).cpu().numpy().astype(np.int8).reshape(-1), fmt='%d')
    np.savetxt(weights_dir + "b_fc1.txt", model.state_dict()['fc1.bias'].cpu().numpy(), fmt='%f')

    # FC2 layer
    w_fc2, scale = quantize_tensor(model.state_dict()['fc2.weight'])
    scales['fc2'] = scale
    np.savetxt(weights_dir + "w_fc2.txt", w_fc2.cpu().numpy().reshape(-1), fmt='%f')
    np.savetxt(weights_dir_int8 + "w_fc2.txt", 
               (w_fc2/scale).cpu().numpy().astype(np.int8).reshape(-1), fmt='%d')
    np.savetxt(weights_dir + "b_fc2.txt", model.state_dict()['fc2.bias'].cpu().numpy(), fmt='%f')

    # FC3 layer
    w_fc3, scale = quantize_tensor(model.state_dict()['fc3.weight'])
    scales['fc3'] = scale
    np.savetxt(weights_dir + "w_fc3.txt", w_fc3.cpu().numpy().reshape(-1), fmt='%f')
    np.savetxt(weights_dir_int8 + "w_fc3.txt", 
               (w_fc3/scale).cpu().numpy().astype(np.int8).reshape(-1), fmt='%d')
    np.savetxt(weights_dir + "b_fc3.txt", model.state_dict()['fc3.bias'].cpu().numpy(), fmt='%f')

    # Lưu scale factors
    with open(weights_dir_int8 + "scales.txt", "w") as f:
        for layer, scale in scales.items():
            f.write(f"{layer}: {scale}\n")

    print("Original weights saved to", weights_dir)
    print("Quantized weights saved to", weights_dir_int8)
    print("Quantization scales:")
    for layer, scale in scales.items():
        print(f"{layer}: {scale}")



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
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    #device = torch.device("cpu")

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
    dataset1 = datasets.MNIST('data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = LeNet5().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    ### Training Phase
    MODEL_FILENAME = "data/lenet5_model.pt"
    if Path(MODEL_FILENAME).exists():
        model.load_state_dict(torch.load(MODEL_FILENAME))
        print('Loaded model from disk')
    else: 
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            test(model, device, test_loader)
            scheduler.step()
            ### Save the model
            if args.save_model:
                torch.save(model.state_dict(), MODEL_FILENAME)
    
    save_weights_to_txt(model)
    
    ### Load the model from lenet5_model.pt file
    print(model)

    ### Inference Phase
    print('Testing...')
    test(model, device, test_loader)
    
if __name__ == '__main__':
   main()



