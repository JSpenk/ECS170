'''
Concrete MethodModule class: a configurable Convolutional Neural Network built with PyTorch.

It reuses the training loop, optimizer/loss options, prediction code and learning-curve
history of Method_MLP, and only replaces the network itself.
'''

# Based on the ECS 170 code template (Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>)
# License: TBD

from local_code.base_class.method import method
from local_code.stage_3_code.Method_MLP import Method_MLP
import torch
from torch import nn


class Method_CNN(Method_MLP):
    data = None

    def __init__(self, mName, mDescription, in_channels=1, input_size=(28, 28), n_classes=10,
                 conv_channels=(32, 64), kernel_size=3, padding=1, stride=1,
                 pool='max', pool_size=2, hidden_dim=128, activation='relu', dropout=0.25,
                 optimizer='adam', learning_rate=1e-3, weight_decay=0.0, loss='cross_entropy',
                 max_epoch=10, batch_size=128, device=None):
        '''
        in_channels / input_size : image layout, e.g. 1 x (28, 28) for MNIST, 3 x (32, 32) for CIFAR
        conv_channels            : one entry per conv block, e.g. (32, 64) = 2 conv blocks
        kernel_size/padding/stride : settings of every conv layer
        pool                     : 'max' | 'avg' | None, applied after every conv block
        hidden_dim               : size of the fully connected layer before the output (0 = none)
        activation               : 'relu' | 'tanh' | 'sigmoid'
        dropout                  : dropout rate before the output layers (0 disables it)
        optimizer / loss         : same options as Method_MLP
        '''
        # same initialization order as Method_MLP
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.in_channels = in_channels
        self.input_size = tuple(input_size)
        self.n_classes = n_classes
        self.conv_channels = list(conv_channels)
        self.hidden_dim = hidden_dim
        self.activation_name = activation
        self.dropout = dropout
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_name = loss
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.softmax_output = False   # raw logits go into the loss
        self.device = torch.device(device) if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        activations = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid}
        pools = {'max': nn.MaxPool2d, 'avg': nn.AvgPool2d}

        # convolution blocks: Conv2d -> activation -> (pooling)
        blocks = []
        prev = in_channels
        for c in self.conv_channels:
            # check here for nn.Conv2d doc: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            blocks.append(nn.Conv2d(prev, c, kernel_size=kernel_size, stride=stride, padding=padding))
            blocks.append(activations[activation]())
            if pool:
                blocks.append(pools[pool](pool_size))
            prev = c
        self.features = nn.Sequential(*blocks)

        # find the flattened feature size by pushing one dummy image through the conv part
        with torch.no_grad():
            flat = self.features(torch.zeros(1, in_channels, *self.input_size)).numel()
        if flat == 0:
            raise ValueError('the conv/pooling settings shrink the image to nothing')

        # classifier: Flatten -> (Linear -> activation -> Dropout) -> Linear
        head = [nn.Flatten()]
        prev = flat
        if hidden_dim > 0:
            head.append(nn.Linear(prev, hidden_dim))
            head.append(activations[activation]())
            prev = hidden_dim
        if dropout > 0:
            head.append(nn.Dropout(dropout))
        head.append(nn.Linear(prev, n_classes))
        self.classifier = nn.Sequential(*head)
        self.to(self.device)

        self.history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'test_acc': []}

    def forward(self, x):
        '''Forward propagation: n x channels x H x W images in, n x n_classes logits out'''
        return self.classifier(self.features(x))