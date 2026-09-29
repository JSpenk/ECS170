'''
Concrete MethodModule class: a configurable Multi-Layer Perceptron built with PyTorch
'''

# Based on the ECS 170 code template (Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>)
# License: TBD

from local_code.base_class.method import method
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics
import torch
from torch import nn
import numpy as np
import time


class Method_MLP(method, nn.Module):
    data = None

    def __init__(self, mName, mDescription, input_dim=784, hidden_dims=(256, 128), n_classes=10,
                 activation='relu', dropout=0.2, optimizer='adam', learning_rate=1e-3,
                 weight_decay=0.0, loss='cross_entropy', max_epoch=30, batch_size=128,
                 softmax_output=False, device=None):
        '''
        input_dim / n_classes : size of the input and output portals (784 pixels -> 10 digits)
        hidden_dims           : one entry per hidden layer, e.g. (256, 128) = 2 hidden layers
        activation            : 'relu' | 'tanh' | 'sigmoid'
        dropout               : dropout rate after every hidden layer (0 disables it)
        optimizer             : 'adam' | 'sgd' (sgd uses momentum 0.9) | 'adagrad'
        loss                  : 'cross_entropy' | 'mse' (MSE against one-hot targets)
        softmax_output        : True reproduces the stage 1 template (softmax layer followed by
                                CrossEntropyLoss); False feeds raw logits into the loss
        '''
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.hidden_dims = list(hidden_dims)
        self.n_classes = n_classes
        self.activation_name = activation
        self.dropout = dropout
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_name = loss
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.softmax_output = softmax_output
        self.device = torch.device(device) if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        activations = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid}
        layers = []
        prev = input_dim
        for h in self.hidden_dims:
            # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
            layers.append(nn.Linear(prev, h))
            layers.append(activations[activation]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.layers = nn.Sequential(*layers)
        self.output_softmax = nn.Softmax(dim=1)
        self.to(self.device)

        # learning curves, filled in during training (one value per epoch)
        self.history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'test_acc': []}

    def forward(self, x):
        '''Forward propagation: returns an n x n_classes tensor (logits, or probabilities
        if softmax_output is set)'''
        out = self.layers(x)
        if self.softmax_output:
            out = self.output_softmax(out)
        return out

    # nn.Module.train(mode) is overridden by the template's train(X, y) below,
    # so switching between training and evaluation mode (needed by Dropout) goes through here
    def _set_mode(self, training):
        nn.Module.train(self, training)

    def _make_optimizer(self):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        if self.optimizer_name == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if self.optimizer_name == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9,
                                   weight_decay=self.weight_decay)
        if self.optimizer_name == 'adagrad':
            return torch.optim.Adagrad(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        raise ValueError('unknown optimizer ' + self.optimizer_name)

    def _loss(self, output, y_true):
        if self.loss_name == 'cross_entropy':
            # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
            return nn.functional.cross_entropy(output, y_true)
        if self.loss_name == 'mse':
            probs = output if self.softmax_output else torch.softmax(output, dim=1)
            target = nn.functional.one_hot(y_true, self.n_classes).float()
            return nn.functional.mse_loss(probs, target)
        raise ValueError('unknown loss ' + self.loss_name)

    @torch.no_grad()
    def _predict(self, X_tensor):
        self._set_mode(False)
        preds = []
        for i in range(0, X_tensor.shape[0], 4096):
            preds.append(self.forward(X_tensor[i:i + 4096]).argmax(dim=1))
        return torch.cat(preds)

    def train(self, X, y, X_monitor=None, y_monitor=None):
        '''Mini-batch gradient descent. X_monitor/y_monitor (the test set) are only used to
        draw the learning curve; they never influence the weights or model selection.'''
        optimizer = self._make_optimizer()
        X = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=self.device)
        y = torch.as_tensor(np.asarray(y), dtype=torch.long, device=self.device)
        if X_monitor is not None:
            X_monitor = torch.as_tensor(np.asarray(X_monitor), dtype=torch.float32, device=self.device)
            y_monitor = np.asarray(y_monitor)
        n = X.shape[0]
        evaluator = Evaluate_Metrics('training evaluator', '')

        for epoch in range(1, self.max_epoch + 1):
            self._set_mode(True)
            perm = torch.randperm(n, device=self.device)
            epoch_loss, correct = 0.0, 0
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                output = self.forward(X[idx])
                train_loss = self._loss(output, y[idx])
                # zero the old gradients, back-propagate the error, update the weights
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                epoch_loss += train_loss.item() * idx.shape[0]
                correct += (output.argmax(dim=1) == y[idx]).sum().item()

            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(epoch_loss / n)
            self.history['train_acc'].append(correct / n)
            msg = 'Epoch: %3d  Loss: %.4f  Train acc: %.4f' % (epoch, epoch_loss / n, correct / n)
            if X_monitor is not None:
                evaluator.data = {'true_y': y_monitor, 'pred_y': self._predict(X_monitor).cpu().numpy()}
                acc = (evaluator.data['true_y'] == evaluator.data['pred_y']).mean()
                self.history['test_acc'].append(float(acc))
                msg += '  Test acc: %.4f' % acc
            print(msg)

    def test(self, X):
        X = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=self.device)
        # instances get the label with the largest output score
        return self._predict(X).cpu().numpy()

    def run(self):
        print('method running...')
        print('--start training...')
        t0 = time.time()
        self.train(self.data['train']['X'], self.data['train']['y'],
                   self.data['test']['X'], self.data['test']['y'])
        self.method_training_time = time.time() - t0
        print('--start testing...')
        t0 = time.time()
        pred_y = self.test(self.data['test']['X'])
        self.method_testing_time = time.time() - t0
        return {'pred_y': pred_y, 'true_y': np.asarray(self.data['test']['y']),
                'history': self.history}
