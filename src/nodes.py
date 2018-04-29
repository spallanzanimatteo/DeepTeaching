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


############
# BASIC NODE
############
class Node(object):
    """Create an operation.

    Each operation takes an input consisting of one ore more operands (which
    are themselves states of other operations) and computes an output state.
    For BackPropagation, the operation should i) accumulate incoming errors,
    ii) then compute how to distribute the corresponding outgoing errors to
    input nodes; if a node contains trainable parameters, it should notify the
    framework that its state can be modified.
    Recurrent cells also require to store an initial internal memory state.
    For BackPropagation Through Time (BPTT), every operation should keep a
    history record of its state, a stack where to push states after each
    timestep of forward propagation and from which to pop them during BPTT.

    Attributes:
        inbound_nodes (:obj:`list` of :obj:`Node`): the operations whose
            states are required to perform the implemented operation.
        outbound_nodes (:obj:`list` of :obj:`Node`): the operations which
            require this operation's state for their computations.
        state (:obj:`ndarray`): the output of the operation.
        incoming_error (:obj:`ndarray`): the error signal collected from
            outbound nodes.
        outgoing_errors (:obj:`dict` of :obj:`ndarray`): the error signals
            to be communicated to inbound nodes.
        is_trainable (:obj:`Bool`): whether the node contains trainable
            parameters.
        initial_state (:obj:`None` or :obj:`ndarray`): the internal memory of
            the operation.
        history (:obj:`list` of :obj:`ndarray`): stack of subsequent states
            of the node.

    Methods:
        push_state: push the current state on self.history stack.
        pop_state: pop last state from self.history stack.
        forward: compute the output of the operation.
        backward: compute the outgoing error signals.

    """

    def __init__(self, inbound_nodes=list()):
        """Create a differentiable operation.

        Args:
            inbound_nodes (:obj:`list` of :obj:`Node`): the operations
                whose states are required to perform the implemented operation.

        """
        self.inbound_nodes = inbound_nodes
        # communicate to inbound nodes that a node has been added on top of them
        for node in self.inbound_nodes:
            node.outbound_nodes.append(self)
        self.outbound_nodes = list()
        self.state = None
        self.incoming_error = None
        self.outgoing_errors = dict()
        self.is_trainable = False
        self.initial_state = None
        self.history = list()

    def push_state(self):
        self.history.append(self.state)

    def pop_state(self):
        self.state = self.history.pop()

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


#############
# INPUT NODES
#############
class Constant(Node):
    """Create an operation to provide a constant operand to the model."""
    def __init__(self, value):
        Node.__init__(self)
        self.state = value

    def forward(self):
        pass

    def backward(self):
        pass


class Placeholder(Node):
    """Create an operation to provide data operands to the model."""
    def __init__(self):
        Node.__init__(self)

    def forward(self, value=None):
        """Provide an operand.

        Args:
            value (:obj:`ndarray`): the data to be fed as operand; it has to
                be a n-dimensional array, with n>=2 (the first dimension is
                the batch size).

        """
        if value is not None:
            self.state = value

    def backward(self):
        pass


class Variable(Node):
    """Create an operation to provide parameter operands to the model."""
    def __init__(self, initial_state):
        """Create an operation to provide parameter operands to the model.

        Args:
            initial_state (:obj:`ndarray`): the initial value of the parameters.

        """
        Node.__init__(self)
        self.state = initial_state
        self.is_trainable = True

    def forward(self):
        pass

    def backward(self):
        # 0) initialize error state
        self.incoming_error = np.zeros_like(self.state)
        # 1) collect error signals from outbound nodes
        for node in self.outbound_nodes:
            self.incoming_error += node.outgoing_errors[self]


############
# LINEAR OPS
############
class Add(Node):
    """Create an element-wise addition operation.

    The operation is defined as $f(x, bias)=x+bias$.

    """
    def __init__(self, x, bias):
        Node.__init__(self, inbound_nodes=[x, bias])

    def forward(self):
        x = self.inbound_nodes[0].state
        b = self.inbound_nodes[1].state
        self.state = x + b

    def backward(self):
        # 1) collect error signals from outbound nodes
        self.incoming_error = np.zeros_like(self.state)
        for node in self.outbound_nodes:
            self.incoming_error += node.outgoing_errors[self]
        # 2) distribute error signals to inbound nodes
        batch_size = self.incoming_error.shape[0]
        self.outgoing_errors[self.inbound_nodes[0]] = self.incoming_error
        bias = self.inbound_nodes[1].state
        if bias.shape == self.incoming_error.shape:
            self.outgoing_errors[self.inbound_nodes[1]] = self.incoming_error
        # 2a) second addend could have been broadcasted over the batch
        # (e.g. bias addition)
        elif bias.shape == self.incoming_error[0, None, :].shape:
            # bias broadcasting happened: accumulate error signals over batch
            self.outgoing_errors[self.inbound_nodes[1]] = np.zeros_like(bias)
            for i in range(batch_size):
                self.outgoing_errors[self.inbound_nodes[1]] += self.incoming_error[i]


class Linear(Node):
    """Create a linear projection operation (MATrix MULtiplication).

    The function is defined as $f(x, W)=xW$.

    """
    def __init__(self, x, w):
        Node.__init__(self, inbound_nodes=[x, w])

    def forward(self):
        x = self.inbound_nodes[0].state
        w = self.inbound_nodes[1].state
        self.state = np.dot(x, w)

    def backward(self):
        # 1) collect error signals from outbound nodes
        self.incoming_error = np.zeros_like(self.state)
        for node in self.outbound_nodes:
            self.incoming_error += node.outgoing_errors[self]
        # 2) distribute error signals to inbound nodes
        batch_size = self.incoming_error.shape[0]
        x = self.inbound_nodes[0].state
        w = self.inbound_nodes[1].state
        # 2a) project the incoming error onto each direction w[i]: w[i] is the
        # mapping of the canonical basis vector $e_i$ of the input space into
        # the output space of the linear projection operation
        self.outgoing_errors[self.inbound_nodes[0]] = np.dot(self.incoming_error, w.T)
        # 2b) accumulate errors due to each example in batch
        self.outgoing_errors[self.inbound_nodes[1]] = np.zeros_like(w)
        for i in range(batch_size):
            patch = x[i, None, :].T * self.incoming_error[i]
            self.outgoing_errors[self.inbound_nodes[1]] += patch


################
# NON-LINEAR OPS
################
class LeakyReLU(Node):
    """Create an element-wise Leaky Rectified Linear Unit (LeakyReLU) operation.

    The function is defined as $f(x)=x if x>0, f(x)=q*x if x<=0$ (with q >= 0).
    It softens the information coming from negative subspaces.

    """
    def __init__(self, q, x):
        Node.__init__(self, inbound_nodes=[x])
        self.q = q

    def forward(self):
        x = self.inbound_nodes[0].state
        mask = x > 0
        mask = mask + self.q * (1-mask)
        self.state = x * mask

    def backward(self):
        # 1) collect error signals from outbound nodes
        self.incoming_error = np.zeros_like(self.state)
        for node in self.outbound_nodes:
            self.incoming_error += node.outgoing_errors[self]
        # 2) distribute error signals to inbound nodes
        x = self.inbound_nodes[0].state
        mask = x > 0
        mask = mask + self.q * (1-mask)
        self.outgoing_errors[self.inbound_nodes[0]] = mask * self.incoming_error


class ReLU(LeakyReLU):
    """Create a Rectified Linear Unit (ReLU) operation.

    The function is defined as $f(x)=x if x>0, f(x)=0 if x<=0$.
    It is a strong version of the LeakyReLU: it unifies all the information
    coming from negative-coordinate subspaces.

    """
    def __init__(self, x):
        LeakyReLU.__init__(self, 0, x)


class Sigmoid(Node):
    """Create a sigmoid operation.

    The function is defined as $f(x)=\frac{e^x}{1+e^x}$. The output is in
    the (0, 1) real interval.

    """
    def __init__(self, x):
        Node.__init__(self, inbound_nodes=[x])

    def forward(self):
        x = self.inbound_nodes[0].state
        e_x = np.exp(x)
        self.state = e_x / (1+e_x)

    def backward(self):
        # 1) collect error signals from outbound nodes
        self.incoming_error = np.zeros_like(self.state)
        for node in self.outbound_nodes:
            self.incoming_error += node.outgoing_errors[self]
        # 2) distribute error signals to inbound nodes
        self.outgoing_errors[self.inbound_nodes[0]] = self.state * (1-self.state) * self.incoming_error


class Softmax(Node):
    """Create a softmax operation.

    The function is defined as $f(x_j)=\frac{e^{x_j}}{\sum_{j=0}^{D-1}e^{x_j}}$,
    i.e. a discrete probability distribution over a finite set of D categories.

    """
    def __init__(self, x):
        Node.__init__(self, inbound_nodes=[x])

    def forward(self):
        x = self.inbound_nodes[0].state
        e_x = np.exp(x)
        self.state = e_x / np.sum(e_x, axis=1)[:, None]

    def backward(self):
        # 1) collect error signals from outbound nodes
        self.incoming_error = np.zeros_like(self.state)
        for node in self.outbound_nodes:
            self.incoming_error += node.outgoing_errors[self]
        # 2) distribute error signals to inbound nodes
        batch_size = self.incoming_error.shape[0]
        self.outgoing_errors[self.inbound_nodes[0]] = np.zeros_like(self.inbound_nodes[0].state)
        for i in range(batch_size):
            prob_i = self.state[i, None, :]
            prob_jacobian = prob_i.T * (np.eye(prob_i.shape[1])-prob_i)
            self.outgoing_errors[self.inbound_nodes[0]][i] = np.dot(self.incoming_error[i], prob_jacobian.T)


class Tanh(Node):
    """Create an hyperbolic tangent operation.

    The function is defined as $f(x)=\frac{e^x - e^{-x}}{e^x + e^{-x}}$.
    The output is in the (-1, 1) real interval.

    """
    def __init__(self, x):
        Node.__init__(self, inbound_nodes=[x])

    def forward(self):
        x = self.inbound_nodes[0].state
        # multiply both numerator and denominator by $e^x$
        e_2x = np.exp(2*x)
        self.state = (e_2x-1) / (e_2x+1)

    def backward(self):
        # 1) collect error signals from outbound nodes
        self.incoming_error = np.zeros_like(self.state)
        for node in self.outbound_nodes:
            self.incoming_error += node.outgoing_errors[self]
        # 2) distribute error signals to inbound nodes
        self.outgoing_errors[self.inbound_nodes[0]] = (1+self.state) * (1-self.state) * self.incoming_error


################
# RISK FUNCTIONS
################
class CCE(Node):
    """Create Categorical Cross Entropy operation.

    The function is defined as
    $R(y_hat, y)=\sum_{j=0}^{D-1}-{y_hat}_{j}\log(y_j)$
    for a D-dimensional target vector $y_hat$.

    """
    def __init__(self, y_hat, y):
        Node.__init__(self, inbound_nodes=[y_hat, y])

    def forward(self):
        y_hat = self.inbound_nodes[0].state
        y = self.inbound_nodes[1].state
        # if no target labels are given, error cannot be measured
        if y_hat is None:
            self.state = 0
        else:
            batch_size = y.shape[0]
            self.state = np.sum(-np.sum(y_hat * np.log(y), axis=1)) / batch_size

    def backward(self):
        # 2) distribute error signals to inbound nodes
        y_hat = self.inbound_nodes[0].state
        y = self.inbound_nodes[1].state
        # if no target labels are given, we have no clue to correct parameters
        if y_hat is None:
            self.outgoing_errors[self.inbound_nodes[1]] = np.zeros_like(y)
        else:
            batch_size = y.shape[0]
            self.outgoing_errors[self.inbound_nodes[1]] = -(y_hat/y) / batch_size


class MSE(Node):
    """Create Mean Squared Error (MSE) operation.

    The function is defined as
    $R(y_hat, y)=\frac{1}{2}\sum_{j=0}^{D-1}({y_hat}_{j}-y_j)^2$
    for a D-dimensional target vector $y_hat$.

    """
    def __init__(self, y_hat, y):
        Node.__init__(self, inbound_nodes=[y_hat, y])

    def forward(self):
        y_hat = self.inbound_nodes[0].state
        y = self.inbound_nodes[1].state
        # if no labels were given, error cannot be measured
        if y_hat is None:
            self.state = 0
        else:
            batch_size = y.shape[0]
            diff = y_hat - y
            self.state = np.sum(np.sum(diff * diff, axis=1) / 2) / batch_size

    def backward(self):
        # 2) distribute error signals to inbound nodes
        y_hat = self.inbound_nodes[0].state
        y = self.inbound_nodes[1].state
        # if no labels were given, we have no clue to correct parameters
        if y_hat is None:
            self.outgoing_errors[self.inbound_nodes[1]] = np.zeros_like(y)
        else:
            batch_size = y.shape[0]
            diff = y_hat - y
            self.outgoing_errors[self.inbound_nodes[1]] = -diff/batch_size


########
# LAYERS
########
class Affine(Node):
    """Create a linear projection operation.

    The function is defined as $f(x, W, b)=xW+b$.

    """
    def __init__(self, x, w, b):
        Node.__init__(self, inbound_nodes=[x, w, b])

    def forward(self):
        x = self.inbound_nodes[0].state
        w = self.inbound_nodes[1].state
        b = self.inbound_nodes[2].state
        self.state = np.dot(x, w) + b

    def backward(self):
        # 1) collect error signals from outbound nodes
        self.incoming_error = np.zeros_like(self.state)
        for node in self.outbound_nodes:
            self.incoming_error += node.outgoing_errors[self]
        # 2) distribute error signals to inbound nodes
        # propagate error through addition
        error_xw = self.incoming_error
        error_b = self.incoming_error
        # propagate error to matmul addend
        batch_size = error_xw.shape[0]
        x = self.inbound_nodes[0].state
        w = self.inbound_nodes[1].state
        self.outgoing_errors[self.inbound_nodes[0]] = np.dot(error_xw, w.T)
        self.outgoing_errors[self.inbound_nodes[1]] = np.zeros_like(w)
        for i in range(batch_size):
            patch = x[i, None, :].T * self.incoming_error[i]
            self.outgoing_errors[self.inbound_nodes[1]] += patch
        # propagate error to bias addend
        batch_size = error_xw.shape[0]
        b = self.inbound_nodes[2].state
        self.outgoing_errors[self.inbound_nodes[2]] = np.zeros_like(b)
        for i in range(batch_size):
            self.outgoing_errors[self.inbound_nodes[2]] += error_b[i]


class RNNCell(Node):
    """Create a simple recurrent cell for RNNs.

    The function is defined as
    $s_t=f(x, W_{xs}, W_{ss}, s)=tanh(xW_{xs}+s_{t-1}W_{ss})$
    where s_t is the internal memory state at time step $t$.

    """
    def __init__(self, x, w_xs, w_ss, initial_state):
        Node.__init__(self, inbound_nodes=[x, w_xs, w_ss])
        self.initial_state = initial_state
        self.inbound_nodes.append(self)
        self.outbound_nodes.append(self)
        self.outgoing_errors[self] = np.zeros_like(initial_state)

    def forward(self):
        x = self.inbound_nodes[0].state
        w_xs = self.inbound_nodes[1].state
        w_ss = self.inbound_nodes[2].state
        # weight information from current event x, weight information from past
        # s_t-1, and merge them to generate present "state of mind" s_t
        logits = np.dot(x, w_xs) + np.dot(self.state, w_ss)
        # tanh activation
        self.state = np.tanh(logits)

    def backward(self):
        # 1) collect error signals from outbound nodes
        self.incoming_error = np.zeros_like(self.state)
        for node in self.outbound_nodes:
            self.incoming_error += node.outgoing_errors[self]
        # 2) distribute error signals to inbound nodes
        # propagate error through tanh
        error_tanh = (1+self.state) * (1-self.state) * self.incoming_error
        # propagate error through addition
        error_xwxs = error_tanh
        error_swss = error_tanh
        # propagate error to x and w_xs
        batch_size = error_xwxs.shape[0]
        x = self.inbound_nodes[0].state
        w_xs = self.inbound_nodes[1].state
        self.outgoing_errors[self.inbound_nodes[0]] = np.dot(error_xwxs, w_xs.T)
        self.outgoing_errors[self.inbound_nodes[1]] = np.zeros_like(w_xs)
        for i in range(batch_size):
            patch = x[i, None, :].T * error_xwxs[i]
            self.outgoing_errors[self.inbound_nodes[1]] += patch
        # propagate error to self and w_ss
        batch_size = error_swss.shape[0]
        wss = self.inbound_nodes[2].state
        self.outgoing_errors[self.inbound_nodes[2]] = np.zeros_like(wss)
        for i in range(batch_size):
            patch = self.state[i, None, :].T * error_swss[i]
            self.outgoing_errors[self.inbound_nodes[2]] += patch
        self.outgoing_errors[self] = np.dot(error_swss, wss.T)
