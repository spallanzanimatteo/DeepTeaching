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
from data_management import get_batches_bptt
from network_management import *
from nodes import Placeholder, Variable, Linear, RNNCell, MSE
from trainers import SGD


class RecurrentModel():
    def __init__(self):
        self.__name__ = 'BloodPressure'
        self.learning_rate = 0.0001
        self.num_epochs = 10
        self.batch_size = 8
        self.time_steps = 10
        self.input_nodes = list()
        self.inference_graph = self._build_inference_graph()
        self.training_graph = self._build_training_graph()

    def _build_inference_graph(self):
        # input layer
        self.X = Placeholder()
        # RNN cell (hidden layer)
        # batch size is required here to setup proper initial state for RNN cell
        num_hidden = 32
        W_XS = Variable(np.random.randn(2, num_hidden))
        W_SS = Variable(np.random.randn(num_hidden, num_hidden))
        initial_hidden_state = np.zeros((self.batch_size, num_hidden))
        self.RNN = RNNCell(self.X, W_XS, W_SS, initial_hidden_state)
        # output layer
        W_SY = Variable(np.random.randn(num_hidden, 1))
        self.y = Linear(self.RNN, W_SY)
        # build inference graph
        self.input_nodes.extend([self.X, W_XS, W_SS, W_SY])
        graph = get_graph_flow(self.input_nodes)

        return graph

    def _build_training_graph(self):
        self.Y_hat = Placeholder()
        self.loss = MSE(self.Y_hat, self.y)
        self.input_nodes.extend([self.Y_hat])
        # add Trainer
        parameters = get_parameters_nodes(self.input_nodes)
        # build training graph
        self.trainer = SGD(parameters, learning_rate=self.learning_rate)
        graph = get_graph_flow(self.input_nodes)

        return graph

    def _load_data(self):
        # load blood pressure data
        if sys.platform == 'win32':
            dataset = pd.read_csv(
                '\\'.join(['.', 'data', 'blood_pressure.txt']), sep=';')
        elif sys.platform == 'linux':
            dataset = pd.read_csv(
                '/'.join(['.', 'data', 'blood_pressure.txt']), sep=';')
        X = dataset.values[:, (1, 3)]
        Y_hat = dataset.values[:, 2][:, None]
        # normalize data
        X = scale(X, axis=0)
        Y_hat = scale(Y_hat, axis=0)
        # create training and validation batches
        batches = get_batches_bptt(X, Y_hat, ts=self.time_steps, bs=self.batch_size)
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
                batch_error = 0.0
                load_initial_states(self.training_graph)
                for t in range(self.time_steps):
                    self.X.forward(value=x[t])
                    self.Y_hat.forward(value=y_hat[t])
                    forward_prop(self.training_graph)
                    push_graph_state(self.training_graph)
                    batch_error += self.loss.state
                save_last_states(self.training_graph)
                train_error += batch_error
                # backward pass
                # compute and accumulate gradients over each unfolded timestep
                for t in range(self.time_steps)[::-1]:
                    pop_graph_state(self.training_graph)
                    backward_prop(self.training_graph)
                    self.trainer.update_gradients()
                # apply corrections
                self.trainer.apply_gradients()
            # store average loss on training batches
            tr_errors.append(train_error/(len(train_batches)*self.time_steps))
            # validation
            valid_error = 0.0
            for x, y_hat in valid_batches:
                batch_error = 0.0
                load_initial_states(self.training_graph)
                for t in range(self.time_steps):
                    self.X.forward(value=x[t])
                    self.Y_hat.forward(value=y_hat[t])
                    forward_prop(self.training_graph)
                    batch_error += self.loss.state
                valid_error += batch_error
            # store average loss on validation batches
            val_errors.append(valid_error/(len(valid_batches)*self.time_steps))
            print("Epoch {:2d} - Loss: {:6.2f}".format(i_epoch+1, val_errors[-1]))
        plt.plot(range(self.num_epochs), tr_errors, label='Training')
        plt.plot(range(self.num_epochs), val_errors, label='Validation')
        plt.legend()
        plt.show()

    def infer(self, x):
        # set RNNCell state to inference mode: only a single state is mantained
        if self.RNN.state.shape[0] is not 1:
            self.RNN.state = self.RNN.state[0, None, :]
        self.X.forward(value=x)
        forward_prop(self.inference_graph)

        return self.y.state


######################
# TEST RECURRENT MODEL
######################
if __name__ == '__main__':
    rec_model = RecurrentModel()
    rec_model.train()
