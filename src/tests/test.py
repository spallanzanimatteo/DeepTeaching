"""
BSD 2-Clause License

Copyright (c) 2018, Matteo Spallanzani
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale

sys.path.insert(0, '..')
from data_management import get_batches
from network_management import *
from nodes import Placeholder, Variable, Matmul, Add, ReLU, MSE
from trainers import SGDWithMomentum


class FeedforwardModel():
    def __init__(self):
        self.__name__ = 'BostonHousingPrices'
        self.learning_rate = 0.001
        self.num_epochs = 40
        self.batch_size = 16
        self._build()

    def _build(self):
        # data nodes
        self.X = Placeholder()
        self.Y_hat = Placeholder()
        # hidden layer
        num_hidden = 20
        W1 = Variable(np.random.randn(3, num_hidden))
        B1 = Variable(np.zeros((1, num_hidden)))
        h = Add(Matmul(self.X, W1), B1)
        act_h = ReLU(h)
        # linear transformation
        W2 = Variable(np.random.randn(num_hidden, 1))
        B2 = Variable(np.zeros((1, 1)))
        self.y = Add(Matmul(act_h, W2), B2)
        self.loss = MSE(self.Y_hat, self.y)
        # set up graph and trainer
        input_nodes = [self.X, self.Y_hat, W1, B1, W2, B2]
        self.graph = get_graph_flow(input_nodes)
        parameters = get_parameters_nodes(self.graph)
        self.trainer = SGDWithMomentum(parameters, learning_rate=self.learning_rate)

    def _load_data(self):
        # load Boston housing prices data
        if sys.platform == 'win32':
            dataset = pd.read_csv(
                '\\'.join(['.', 'data', 'boston_housing.txt']))
        elif sys.platform == 'linux':
            dataset = pd.read_csv(
                '/'.join(['.', 'data', 'boston_housing.txt']))
        x = dataset.values[:, 0:3]
        y_hat = dataset.values[:, 3][:, None]
        # normalize data
        x = scale(x, axis=0)
        y_hat = scale(y_hat, axis=0)
        # create training and validation batches
        batches = get_batches(x, y_hat, bs=self.batch_size)
        num_batches = len(batches)
        val = 0.10
        num_val_batches = int(val * num_batches)
        train_batches = batches[:-num_val_batches]
        valid_batches = batches[-num_val_batches:]

        return train_batches, valid_batches

    def train(self):
        train_batches, valid_batches = self._load_data()
        # training/validation statistics
        tr_errors = list()
        val_errors = list()
        for i_epoch in range(self.num_epochs):
            # training
            train_error = 0.0
            for x, y_hat in train_batches:
                # forward pass
                self.X.forward(value=x)
                self.Y_hat.forward(value=y_hat)
                forward_prop(self.graph)
                train_error += self.loss.state
                # backward pass
                # compute gradients
                backward_prop(self.graph)
                self.trainer.update_gradients()
                # apply corrections
                self.trainer.apply_gradients()
            tr_errors.append(train_error/len(train_batches))
            # validation
            valid_error = 0
            for x, y_hat in valid_batches:
                self.X.forward(value=x)
                self.Y_hat.forward(value=y_hat)
                forward_prop(self.graph)
                valid_error += self.loss.state
            val_errors.append(valid_error/len(valid_batches))
            print("Epoch {:2d} - Loss: {:4.2f}".format(i_epoch+1, val_errors[-1]))
        plt.plot(range(self.num_epochs), tr_errors)
        plt.plot(range(self.num_epochs), val_errors)
        plt.show()


########################
# TEST FEEDFORWARD MODEL
########################
if __name__ == '__main__':
    ff_model = FeedforwardModel()
    ff_model.train()
