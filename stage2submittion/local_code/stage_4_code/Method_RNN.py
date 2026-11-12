'''
Base MethodModule class for all models and frameworks
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD


from local_code.base_class.method import method
from local_code.stage_4_code.Method_MLP import Method_MLP
import torch
from torch import nn
import numpy as np
import re
import time

from collections import Counter
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

class METHOD_RNN(Method_MLP):

    data = None

    def __init__(self, mName, mDescription, vocab_size=10000,
                 embedding_dim=128, max_length=300, n_classes=2,
                 rnn_type='rnn', rnn_hidden=128, num_layers=1,
                 bidirectional=False, rnn_dropout=0.0,
                 hidden_dim=128, activation='relu', dropout=0.25,
                 optimizer='adam', learning_rate=1e-3,
                 weight_decay=0.0, loss='cross_entropy',
                 max_epoch=10, batch_size=128, device=None):

        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.n_classes = n_classes
        self.rnn_type = rnn_type.lower()
        self.rnn_hidden = rnn_hidden
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn_dropout = rnn_dropout
        self.hidden_dim = hidden_dim
        self.activation_name = activation
        self.dropout = dropout
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_name = loss
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.softmax_output = False

        self.device = torch.device(device) if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Vocabulary: 0 = padding, 1 = unknown word
        self.word_to_id = {'<PAD>': 0, '<UNK>': 1}

        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=0
        )

        # Select recurrent architecture
        rnns = {
            'rnn': nn.RNN,
            'lstm': nn.LSTM,
            'gru': nn.GRU
        }

        if self.rnn_type not in rnns:
            raise ValueError('unknown RNN type ' + self.rnn_type)

        self.rnn = rnns[self.rnn_type](input_size=embedding_dim, hidden_size=rnn_hidden, num_layers=num_layers,batch_first=True, bidirectional=bidirectional, dropout=rnn_dropout if num_layers > 1 else 0.0)

        activations = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid}

        if activation not in activations:
            raise ValueError('unknown activation ' + activation)

        directions = 2 if bidirectional else 1
        prev = rnn_hidden * directions

        head = []

        if hidden_dim > 0:
            head.append(nn.Linear(prev, hidden_dim))
            head.append(activations[activation]())
            prev = hidden_dim

        if dropout > 0:
            head.append(nn.Dropout(dropout))

        head.append(nn.Linear(prev, n_classes))

        self.classifier = nn.Sequential(*head)
        self.to(self.device)
        self.history = {'epoch': [],'train_loss': [], 'train_acc': [],'test_acc': []}

    @staticmethod
    def _clean_text(text):
        text = text.lower()
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _tokenize(self, text):
        return re.findall(r"\b\w+\b|[^\w\s]", self._clean_text(text))

    def _build_vocabulary(self, reviews):
        counts = Counter()

        for review in reviews:
            counts.update(self._tokenize(review))

        most_common = counts.most_common(self.vocab_size - 2)

        self.word_to_id = {'<PAD>': 0, '<UNK>': 1}

        for word, _ in most_common:
            self.word_to_id[word] = len(self.word_to_id)

    def _encode_reviews(self, reviews):
        sequences = []
        lengths = []

        for review in reviews:
            tokens = self._tokenize(review)

            ids = [
                self.word_to_id.get(token, 1)
                for token in tokens
            ]

            ids = ids[:self.max_length]
            lengths.append(max(1, len(ids)))

            if len(ids) == 0:
                ids = [1]

            sequences.append(ids)

        X = np.zeros((len(sequences), self.max_length), dtype=np.int64)

        for i, sequence in enumerate(sequences):
            X[i, :len(sequence)] = sequence

        return X, np.asarray(lengths, dtype=np.int64)

    # Forward propagation
    def forward(self, x, lengths=None):

        embedded = self.embedding(x)

        if lengths is not None:
            lengths_cpu = torch.as_tensor(lengths, dtype=torch.long, device='cpu')
            packed = pack_padded_sequence(embedded, lengths_cpu, batch_first=True, enforce_sorted=False)
            _, h = self.rnn(packed)
        else:
            _, h = self.rnn(embedded)
        if self.rnn_type == 'lstm':
            h = h[0]
        if self.bidirectional:
            last = torch.cat([h[-2], h[-1]], dim=1)
        else:
            last = h[-1]

        return self.classifier(last)

    @torch.no_grad()
    def _predict(self, X_tensor, lengths):

        self._set_mode(False)

        predictions = []

        for i in range(0, X_tensor.shape[0], 4096):
            batch_x = X_tensor[i:i + 4096]
            batch_lengths = lengths[i:i + 4096]
            output = self.forward(batch_x, batch_lengths)
            predictions.append(output.argmax(dim=1))

        return torch.cat(predictions)

    def train(self, X, y, lengths, X_monitor=None,
              y_monitor=None, monitor_lengths=None):

        optimizer = self._make_optimizer()

        X = torch.as_tensor(
            X, dtype=torch.long, device=self.device
        )

        y = torch.as_tensor(
            np.asarray(y), dtype=torch.long, device=self.device
        )

        lengths = torch.as_tensor(
            lengths, dtype=torch.long
        )

        if X_monitor is not None:
            X_monitor = torch.as_tensor(
                X_monitor, dtype=torch.long, device=self.device
            )

            y_monitor = np.asarray(y_monitor)

            monitor_lengths = torch.as_tensor(
                monitor_lengths, dtype=torch.long
            )

        dataset = TensorDataset(X, y, lengths)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        n = X.shape[0]

        for epoch in range(1, self.max_epoch + 1):

            self._set_mode(True)

            epoch_loss = 0.0
            correct = 0

            for batch_x, batch_y, batch_lengths in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                output = self.forward(batch_x, batch_lengths)
                train_loss = self._loss(output, batch_y)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                epoch_loss += train_loss.item() * batch_x.shape[0]
                correct += (
                    output.argmax(dim=1) == batch_y
                ).sum().item()

            train_loss_value = epoch_loss / n
            train_acc = correct / n

            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(train_loss_value)
            self.history['train_acc'].append(train_acc)

            msg = (
                'Epoch: %3d  Loss: %.4f  Train acc: %.4f'
                % (epoch, train_loss_value, train_acc)
            )

            if X_monitor is not None:

                predictions = self._predict(
                    X_monitor, monitor_lengths
                ).cpu().numpy()

                acc = (y_monitor == predictions).mean()
                self.history['test_acc'].append(float(acc))
                msg += '  Test acc: %.4f' % acc

            print(msg)

    def test(self, X, lengths):

        X = torch.as_tensor(
            X, dtype=torch.long, device=self.device
        )

        predictions = self._predict(X, lengths)

        return predictions.cpu().numpy()

    def run(self):
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'test_acc': []
        }

        print('method running...')
        print('--preprocessing text...')

        train_reviews = self.data['train']['X']
        train_labels = self.data['train']['y']

        test_reviews = self.data['test']['X']
        test_labels = self.data['test']['y']

        self._build_vocabulary(train_reviews)
        X_train, train_lengths = self._encode_reviews(train_reviews)
        X_test, test_lengths = self._encode_reviews(test_reviews)

        print('Vocabulary size:', len(self.word_to_id))
        print('--start training...')

        t0 = time.time()
        self.train(X_train, train_labels,train_lengths,
                   X_test,test_labels, test_lengths)
        self.method_training_time = time.time() - t0

        print('--start testing...')

        t0 = time.time()
        pred_y = self.test(X_test, test_lengths)
        self.method_testing_time = time.time() - t0

        return {'pred_y': pred_y, 'true_y': np.asarray(test_labels), 'history': self.history}
