import tensorflow as tf


def update_symmetry_edge_features(node_features, edges, edge_features, message_fn):

    """
    Pass messages between nodes and sum the incoming messages at each node.
    Implements equation 1 and 2 in the paper, i.e. m_{.j}^t &= \\sum_{i \\in N(j)} f(h_i^{t-1}, h_j^{t-1})
    : nodes_features: (n_nodes, n_features) tensor of node hidden states.
    : edges         : (n_edges, 2) tensor of indices (i, j) indicating an edge from nodes[i] to nodes[j].
    : edge_features : features for each edge. Set to zero if the edges don't have features.
    : message_fn    : message function, or convolution filter, takes input (n_edges, 2*n_features+n_output) and returns
    output (n_edges, n_output)
    : return        : (n_nodes, n_output) Sum of messages arriving at each node.
    """
    n_nodes = tf.shape(node_features)[0]
    n_features = tf.shape(node_features)[1]

    reshaped         = tf.reshape(tf.gather(node_features, edges), (-1, 2 * n_features))
    symmetric      = tf.math.multiply(0.5, tf.math.add(reshaped[:, 0:n_features], reshaped[:, n_features:2*n_features]))
    asymmetric     = tf.math.multiply(0.5, tf.math.abs(tf.math.subtract(reshaped[:, 0:n_features],
                                                                        reshaped[:, n_features:2*n_features])))
    messages       = message_fn(tf.concat([symmetric, asymmetric, edge_features], axis=1))  # n_edges, n_output

    updates = tf.math.add(tf.math.unsorted_segment_sum(messages, edges[:, 0], n_nodes), tf.math.unsorted_segment_sum(
        messages, edges[:, 1], n_nodes))  # aggregate the edge messages around every node

    return messages, updates


def update_node_features(node_features, grad_P1, message_fn):
    """
    Pass messages from edges to the nodes
    : node_features: (n_nodes, n_features) tensor of node hidden states.
    : edge_features: 1st-order gradient (n_nodes, 2, n_features)
    : message_fn   : message function, takes input (n_nodes, n_features+n_output)
     and returns output (n_nodes, n_features)
    """
    message_input = tf.keras.layers.concatenate([node_features, grad_P1], axis=1)
    updated       = message_fn(message_input)

    return updated


class EdgeSmoothing(tf.keras.layers.Layer):
    def __init__(self):
        super(EdgeSmoothing, self).__init__()
    
    def call(self, to_concat, node_features, edges, count):
        n_nodes        = tf.shape(node_features)[0]
        flow_on_edge   = tf.math.reduce_mean(tf.gather(node_features, edges), 1)  # n_edges, n_features
        aggre_flow     = tf.math.add(tf.math.unsorted_segment_sum(flow_on_edge[:, :], edges[:, 0], n_nodes),
                                 tf.math.unsorted_segment_sum(flow_on_edge[:, :], edges[:, 1], n_nodes))

        return tf.keras.layers.concatenate([to_concat, tf.math.divide(aggre_flow, count)], axis=1)


class MLP(tf.keras.layers.Layer):
    """
    Message passing function used for graph convolution layer. It is used both for edge convolution and node convolution
    The implemented MLP has one single hidden layer with ReLu activation.  The output layer is a linear layer.
    : hidden_nodes : number of neurons in the hidden layer.
    : output_dim   : dimension of the output
    : initializer  : method to initialize the trainable weights and bias. When MLP is embedded in a graph
    convolution layer, it inherits the layer's initializer
    """
    
    def __init__(self, hidden_nodes, output_dim, initializer):

        super(MLP, self).__init__()
        self.hidden_nodes    = hidden_nodes
        self.output_dim      = output_dim
        self.initializer     = initializer

    def build(self, input_shape):

        self.weights_1 = self.add_weight(name='hid_layer', shape=(input_shape[-1]+1, self.hidden_nodes),
                                         initializer=self.initializer, trainable=True)
        self.weights_2 = self.add_weight(name='out_layer', shape=(self.hidden_nodes+1, self.output_dim),
                                         initializer=self.initializer, trainable=True)

    def call(self, inputs):
        
        x = tf.math.add(tf.linalg.matmul(inputs, self.weights_1[:-1, :]), self.weights_1[-1, :])
        hidden_values = tf.math.multiply(x, tf.nn.sigmoid(x))
        
        y = tf.math.add(tf.linalg.matmul(hidden_values, self.weights_2[:-1, :]), self.weights_2[-1, :])
        out_layer = tf.math.multiply(y, tf.nn.sigmoid(y))
        
        return out_layer


class InvariantEdgeConv(tf.keras.layers.Layer):
    """
    Graph convolution adapted from the implementation in GraphNets@DeepMind ( https://arxiv.org/abs/1806.01261 ).
    Node features on the two nodes of an edge, along with the old features on this edge, are used to update
    the edge features.
    Updated edges features around a nodes are summed, along with the old features on this node, are used to update
    the node features
    :
    """

    def __init__(self, edge_feature_dim, num_filters, mlp_width, initializer):

        super(InvariantEdgeConv, self).__init__()

        self.edge_feat_dim     = edge_feature_dim
        self.num_filters       = num_filters
        self.initializer       = initializer
        self.mlp_width         = mlp_width
        self.message_fn_edge   = MLP(self.mlp_width, self.edge_feat_dim, self.initializer)
        self.message_fn_node   = MLP(self.mlp_width, self.num_filters, self.initializer)

    def call(self, node_features, edge_features, edges):

        updated_edge_features, contribution_edges     = update_symmetry_edge_features(node_features, edges,
                                                                                      edge_features,
                                                                                      self.message_fn_edge)
        updated_node_features                         = update_node_features(node_features, contribution_edges,
                                                                             self.message_fn_node)

        return updated_node_features, updated_edge_features

class InvariantEdgeModel(tf.keras.Model):
    def __init__(self, edge_feature_dims, num_filters, depth, mlp_width, initializer):
        super(InvariantEdgeModel, self).__init__()

        self.edge_feat_dims = edge_feature_dims
        self.num_filters    = num_filters
        self.depth          = depth
        self.mlp_width      = mlp_width
        self.initializer    = initializer

        self.edge_convs = [
            InvariantEdgeConv(self.edge_feat_dims[i], self.num_filters[i], self.mlp_width, self.initializer)
            for i in range(depth)  # sequential layers of InvariantEdgeConv
        ]

        self.smoothing_layers = [
            EdgeSmoothing() for _ in range(depth)  # sequential layers of EdgeSmoothing
        ]
       
        self.out_layer = tf.keras.layers.Dense(3, activation=None, kernel_initializer=self.initializer)
    
    def call(self, node_input, edges, edge_input, smoothing_weights):

        new_node_features = node_input
        new_edge_features = edge_input

        # Iterate through the graph convolution layers and smoothing layers
        for i in range(self.depth):
            new_node_features, new_edge_features = self.edge_convs[i](new_node_features, new_edge_features, edges)
            new_node_features = self.smoothing_layers[i](node_input[:, 0:2], new_node_features, edges, smoothing_weights)

        # output dense layer, without nonlinear activation
        node_outputs                             = self.out_layer(new_node_feature[:, 0:])

        return node_outputs
