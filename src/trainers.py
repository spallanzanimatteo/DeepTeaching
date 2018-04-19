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


import numpy as np


class BaseTrainer():
    """Create a trainer to monitor and update models :obj:`Variable` nodes.

    This object is linked to all the :obj:`Variable` nodes of a given graph, and
    is in charge of the bookkeeping of the gradients and the application of the
    training numerical optimization algorithm.

    Attributes:
        gradients (:obj:`dict` of :obj:`ndarray`s): the gradients of each of the
            model's parameters, computed during last backpropagation pass.
        learning_rate (:obj:`float`): the step length of the numerical
            optimization algorithm.

    Methods:
        _reset_gradients: reset gradients accumulators to zero.
        update_gradients: accumulate each parameter's `incoming_error` into the
            corresponding gradient accumulator.
        apply_gradients: modify each parameter's `state` according to the
            corresponding gradient and optimization algorithm.

    """
    def __init__(self, parameters, learning_rate):
        """Create the trainer and link it to models variables."""
        self.gradients = dict()
        for node in parameters:
            self.gradients[node] = np.zeros_like(node.state)
        self.learning_rate = learning_rate

    def _reset_gradients(self):
        for (node, grad) in self.gradients.items():
            grad *= 0.0

    def update_gradients(self):
        for (node, grad) in self.gradients.items():
            grad += node.incoming_error

    def apply_gradients(self):
        pass


class SGD(BaseTrainer):
    """Vanilla Stochastic Gradient Descent (SGD) optimizer."""
    def __init__(self, parameters, learning_rate=0.001):
        BaseTrainer.__init__(self, parameters, learning_rate)

    def apply_gradients(self):
        for (node, grad) in self.gradients.items():
            node.state -= self.learning_rate*grad
        self._reset_gradients()


class SGDWithMomentum(SGD):
    """Stochastic Gradient Descent with momentum.

    Attributes:
        beta (:obj:`float`): the coefficient of the geometric mean held between
            past and current gradients.
        moments (:obj:`dict` of :obj:`ndarray`s): the geometric mean of past
            gradients.

    """
    def __init__(self, parameters, learning_rate=0.001, beta=0.9):
        SGD.__init__(self, parameters, learning_rate)
        self.beta = beta
        self.moments = dict()
        for node in parameters:
            self.moments[node] = np.zeros_like(node.state)

    def apply_gradients(self):
        for (node, grad) in self.gradients.items():
            true_grad = (1.0-self.beta)*grad + self.beta*self.moments[node]
            node.state -= self.learning_rate*true_grad
            self.moments[node] = true_grad
        self._reset_gradients()
