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


##################
# GRAPH MANAGEMENT
##################
def get_connectivity(input_nodes):
    """Create a description of the connections of each node in the graph.

    Recurrent connections (i.e. connections of a node with itself) are excluded.

    Args:
        input_nodes (:obj:`list` of :obj:`Node`): the input operations of
            the model.

    Returns:
        graph (:obj:`dict` of :obj:`dict` of :obj:`set`): a description of the
            graph's connectivity in terms of inbound-outbound nodes of each
            node.

    """

    graph = dict()
    nodes = input_nodes.copy()
    while len(nodes) != 0:
        # select a node
        current_node = nodes.pop(0)
        # if no information has been collected yet, set up dict entry
        if current_node not in graph:
            graph[current_node] = {'inbound': set(), 'outbound': set()}
        # scroll through current node's outbound nodes
        for node in current_node.outbound_nodes:
            # skip recurrent connections (for RNN cells)
            if node == current_node:
                continue
            # if no information has been collected yet, set up dict entry
            if node not in graph:
                nodes.append(node)
                graph[node] = {'inbound': set(), 'outbound': set()}
            # add reciprocal connectivity information
            graph[current_node]['outbound'].add(node)
            graph[node]['inbound'].add(current_node)

    return graph


def topological_sort(input_nodes, graph):
    """Get a consistent sequence of operations on the given graph.

    Args:
        input_nodes (:obj:`list` of :obj:`Node`): the input operations of
            the model.
        graph (:obj:`dict` of :obj:`dict` of :obj:`set`): a description of the
            graph's connectivity.

    Returns:
        sorted_nodes (:obj:`list` of :obj:`Node`): a sequence of operations
            that ensures computational consistency of the model.

    """

    sorted_nodes = list()
    unlocked_nodes = input_nodes.copy()
    while len(unlocked_nodes) != 0:
        # select an inbound-free node and add it the sorted list
        # (it is ok for computation since all "requirement" nodes are available)
        current_node = unlocked_nodes.pop(0)
        sorted_nodes.append(current_node)
        current_outbound = graph[current_node]['outbound']
        if current_outbound is None:
            # dead end reached
            continue
        for node in graph[current_node]['outbound']:
            # free the outbound node from requiring current node
            graph[node]['inbound'].remove(current_node)
            # if the outbound node has no more requirements to be fulfilled,
            # unlock it
            if len(graph[node]['inbound']) == 0:
                unlocked_nodes.append(node)

    return sorted_nodes


def get_graph_flow(input_nodes):
    """Build the network graph.

    A wrapper function to automate model build.

     Args:
        input_nodes (:obj:`list` of :obj:`Node`): the input operations of
            the graph.

     Returns:
        requirements_chain (:obj:`list` of :obj:`Node`): a sequence of
            operations that ensures computational consistency of the model.

   """

    connectivity = get_connectivity(input_nodes)
    requirements_chain = topological_sort(input_nodes, connectivity)

    return requirements_chain


def get_parameters_nodes(input_nodes):
    """Find operations containing the parameters of the model.

     Args:
        input_nodes (:obj:`list` of :obj:`Node`): the input operations of the
            model.

     Returns:
        parameters (:obj:`list` of :obj:`Node`): the operations containing
            the parameters of the model.

    """

    parameters = list()
    for node in input_nodes:
        if node.is_trainable:
            parameters.append(node)

    return parameters


def push_graph_state(graph):
    """Take a snapshot of current graph state.

    Args:
        graph (:obj:`list` of :obj:`Node`): the operations that should push
            their current state on their history stack.

    """
    for node in graph:
        if not node.is_trainable:
            node.push_state()


def pop_graph_state(graph):
    """Load last known snapshot of graph history.

    Args:
        graph (:obj:`list` of :obj:`Node`): the operations that should pop
            last entry from their history stack.

    """
    for node in graph:
        if not node.is_trainable:
            node.pop_state()


def save_last_states(graph, stateful=True):
    """Store last state of RNNs' recurrent cells.

    When a RNN forward pass is completed, the last known internal state should
    be saved. In fact, the backward pass pops and then **discards** each
    timestep state; but when the RNN is stateful, the last known state of the
    current batch will be reused.

    Args:
        graph (:obj:`list` of :obj:`Node`): the complete sequence of
            operations in the model.
        stateful (:obj:`Bool`): whether the last memory state after the
            complete batch should be passed to the next batch (stateful RNNs)
            or if the internal memory should be reset (stateless RNNs).

    """
    for node in graph:
        # just for nodes that process sequences
        if node.initial_state is not None:
            if stateful:
                # remember last state of the node
                node.initial_state = node.state
            else:
                # reset state
                node.initial_state *= 0


def load_initial_states(graph):
    """Initialize states of RNNs' recurrent cells.

    Args:
        graph (:obj:`list` of :obj:`Node`): the complete sequence of
            operations in the model.

    """
    for node in graph:
        if node.initial_state is not None:
            node.state = node.initial_state


##################
#  FLOW MANAGEMENT
##################
def forward_prop(requirements_chain):
    """Push the current inputs through the whole model.

    Consistently complete the systolic sequence of operations.

    Args:
        requirements_chain (:obj:`list` of :obj:`Node`): a sequence of
            operations that ensures computational consistency of the model.

    """
    for node in requirements_chain:
        node.forward()


def backward_prop(requirements_chain):
    """Propagate error signals backward through the model (but do not apply
    corrections).

    Args:
        requirements_chain (:obj:`list` of :obj:`Node`): a sequence of
            operations that ensures computational consistency of the model.

    """
    reverse_chain = requirements_chain[::-1]
    for node in reverse_chain:
        node.backward()
