# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 13:56:29 2019

@author: divya
"""
from OOAD_CNN import *
from OOAD_training import *
import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn as nn

def main():   
    # Training settings
    parser = argparse.ArgumentParser(description='CNN')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                       help='training batch size, default=64')
    parser.add_argument('--test_batch_size', type=int, default=100, metavar='N',
                       help='testing batch size, default:100')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                       help='number of training epochs, default: 5')
    parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                       help='learning rate, default: 0.005')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                       help='SGD momentum, default: 0.5')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                       help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                       help='random seed, default: 1234')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                       help='batches between logging training status')
    parser.add_argument('--save_model', action='store_true', default=True,
                       help='Save current Model')   

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    kwargs = {'num_workers': 2} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../DATA', train=True, download=True,
                       transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))])),
                       batch_size=args.batch_size, shuffle=True, **kwargs)
    
    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../DATA', train=False, 
                       transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))])),
                       batch_size=args.test_batch_size, shuffle=True, **kwargs)
    
    num_classes=10
    model=ConvNet(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    run=trainTest()
    
    for epoch in range(1, args.epochs + 1):
        run.train(args, model, device, train_loader, optimizer, criterion, epoch)
        run.test(args, model, device, criterion,test_loader)

    if (args.save_model):
        torch.save(model.state_dict(),"../MODEL/mnistCNN.pt")

if __name__ == '__main__':
    main()