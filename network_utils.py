import tensorflow as tf

def update_symmetry_edge_features(node_features, edges, edge_features, message_fn):

    """
    Pass messages between nodes and sum the incoming messages at each node.
    Implements equation 1 and 2 in the paper, i.e. m_{.j}^t &= \sum_{i \in N(j)} f(h_i^{t-1}, h_j^{t-1})
    : nodes_features: (n_nodes, n_features) tensor of node hidden states.
    : edges         : (n_edges, 2) tensor of indices (i, j) indicating an edge from nodes[i] to nodes[j].
    : edge_features : features for each edge. Set to zero if the edges don't have features.
    : message_fn    : message function, or convolution filter, takes input (n_edges, 2*n_features+n_output) and returns output (n_edges, n_output)
    : return        : (n_nodes, n_output) Sum of messages arriving at each node.
    """
	
    n_nodes = tf.shape(node_features)[0]
    n_features = tf.shape(node_features)[1]

    reshaped         = tf.reshape(tf.gather(node_features, edges), (-1, 2 * n_features))
    symmetric      = tf.math.multiply(0.5, tf.math.add(reshaped[:, 0:n_features], reshaped[:, n_features:2*n_features]))
    asymmetric     = tf.math.multiply(0.5, tf.math.abs(tf.math.subtract(reshaped[:, 0:n_features], reshaped[:, n_features:2*n_features])))
    messages       = message_fn(tf.concat([symmetric, asymmetric, edge_features], axis=1))  # n_edges, n_output

    updates = tf.math.add(tf.math.unsorted_segment_sum(messages, edges[:, 0], n_nodes), tf.math.unsorted_segment_sum(messages, edges[:, 1], n_nodes))##agregate the edge messages around every node

    return messages, updates# n_nodes, n_output

def update_node_features(node_features, grad_P1, message_fn):
    """
    Pass messages from edges to the nodes
    : node_features: (n_nodes, n_features) tensor of node hidden states.
    : edge_features: 1st-order gradient (n_nodes, 2, n_features)
    : message_fn   : message function, takes input (n_nodes, n_features+n_output) and returns output (n_nodes, n_features)
    """
	
    message_input = tf.keras.layers.concatenate([node_features, grad_P1], axis=1)
    updated = message_fn(message_input)
	
    return updated


class edge_smoothing(tf.keras.layers.Layer):
    def __init__(self):
        super(edge_smoothing, self).__init__()
    
    def call(self, to_concat, node_features, edges, count):
        n_nodes        = tf.shape(node_features)[0]
        n_features     = tf.shape(node_features)[1]
        flow_on_edge   = tf.math.reduce_mean(tf.gather(node_features, edges), 1)# n_edges, n_features
        aggre_flow = tf.math.add(tf.math.unsorted_segment_sum(flow_on_edge[:,:], edges[:,0], n_nodes), tf.math.unsorted_segment_sum(flow_on_edge[:,:], edges[:,1], n_nodes))# (n_nodes, n_features)

        return tf.keras.layers.concatenate([to_concat, tf.math.divide(aggre_flow, count)], axis=1)
		
		
class MLP(tf.keras.layers.Layer):
    """
    Message passing function used for graph convolution layer. It is used both for edge convolution and node convolution
    The implemented MLP has one single hidden layer with ReLu activation.  The output layer is a linear layer.
    : hidden_nodes : number of neurons in the hidden layer.
    : output_dim   : dimension of the output
    : initializer  : method to initialize the trainable weights and bias. When MLP is embedded in a graph convolution layer, it inherits the layer's initializer 
    """
    
    def __init__(self, hidden_nodes, output_dim, initializer):

        super(MLP, self).__init__()
        self.hidden_nodes    = hidden_nodes
        self.output_dim      = output_dim
        self.initializer     = initializer

    def build(self, input_shape):

        self.weights_1 = self.add_weight(name='hid_layer',
            shape=(input_shape[-1]+1, self.hidden_nodes),
            initializer=self.initializer,
            trainable=True)
        self.weights_2 = self.add_weight(name='out_layer',
            shape=(self.hidden_nodes+1, self.output_dim),
            initializer=self.initializer,
            trainable=True)

    def call(self, inputs):
        
        x = tf.math.add(tf.linalg.matmul(inputs, self.weights_1[:-1, :]), self.weights_1[-1, :])
        hidden_values = tf.math.multiply(x, tf.nn.sigmoid(x))
        
        y = tf.math.add(tf.linalg.matmul(hidden_values, self.weights_2[:-1, :]), self.weights_2[-1, :])
        out_layer = tf.math.multiply(y, tf.nn.sigmoid(y))
        
        return out_layer


class invariant_edge_conv(tf.keras.layers.Layer):
    """
    Graph convolution adapted from the implementation in GraphNets@DeepMind ( https://arxiv.org/abs/1806.01261 ).
    Node features on the two nodes of an edge, along with the old features on this edge, are used to update the edge features.
    Updated edges features around a nodes are summed, along with the old features on this node, are used to update the node features
    :
    """

    def __init__(self, edge_feature_dim, num_filters, initializer):

        super(invariant_edge_conv, self).__init__()

        self.edge_feat_dim     = edge_feature_dim
        self.num_filters       = num_filters
        self.initializer       = initializer
        self.message_fn_edge = MLP(128, self.edge_feat_dim, self.initializer)
        self.message_fn_node = MLP(128, self.num_filters, self.initializer)
     

    def call(self, node_features, edge_features, edges):

        updated_edge_features, contribution_edges     = update_symmetry_edge_features(node_features, edges, edge_features, self.message_fn_edge)
        updated_node_features                         = update_node_features(node_features, contribution_edges, self.message_fn_node)

        return updated_node_features, updated_edge_features


class invariant_edge_model(tf.keras.Model):
    def __init__(self, edge_feature_dims, num_filters, initializer):
        super(invariant_edge_model, self).__init__()

        self.edge_feat_dims = edge_feature_dims
        self.num_filters    = num_filters
        self.initializer     = initializer

        self.layer0  = invariant_edge_conv(self.edge_feat_dims[0], self.num_filters[0], self.initializer)
        self.layer00 = edge_smoothing()
        
        self.layer1  = invariant_edge_conv(self.edge_feat_dims[1], self.num_filters[1], self.initializer)
        self.layer11 = edge_smoothing()
        
        self.layer2  = invariant_edge_conv(self.edge_feat_dims[2], self.num_filters[2], self.initializer)
        self.layer22 = edge_smoothing()
        
        self.layer3  = invariant_edge_conv(self.edge_feat_dims[3], self.num_filters[3], self.initializer)
        self.layer33 = edge_smoothing()
        
        self.layer4  = invariant_edge_conv(self.edge_feat_dims[4], self.num_filters[4], self.initializer)
        self.layer44 = edge_smoothing()
        
        self.layer5  = invariant_edge_conv(self.edge_feat_dims[5], self.num_filters[5], self.initializer)
        self.layer55 = edge_smoothing()
        
        self.layer6  = invariant_edge_conv(self.edge_feat_dims[6], self.num_filters[6], self.initializer)
        self.layer66 = edge_smoothing()
        
        self.layer7  = invariant_edge_conv(self.edge_feat_dims[7], self.num_filters[7], self.initializer)
        self.layer77 = edge_smoothing()
       
        self.layer8 = tf.keras.layers.Dense(3, activation=None, kernel_initializer=self.initializer)
    
    def call(self, node_input, edges, edge_input, smoothing_weights):


        ## graph convolution
        new_node_features_0, new_edge_features_0 = self.layer0(node_input, edge_input, edges)
		## smoothing plus concatenation
        smoothed_0          = self.layer00(node_input[:,0:2], new_node_features_0, edges, smoothing_weights)

        new_node_features_1, new_edge_features_1 = self.layer1(smoothed_0, new_edge_features_0, edges)
        smoothed_1          = self.layer11(node_input[:,0:2], new_node_features_1, edges, smoothing_weights)

        new_node_features_2, new_edge_features_2 = self.layer2(smoothed_1, new_edge_features_1, edges)
        smoothed_2          = self.layer22(node_input[:,0:2], new_node_features_2, edges, smoothing_weights)

        new_node_features_3, new_edge_features_3 = self.layer3(smoothed_2, new_edge_features_2, edges)
        smoothed_3          = self.layer33(node_input[:,0:2], new_node_features_3, edges, smoothing_weights)

        new_node_features_4, new_edge_features_4 = self.layer4(smoothed_3, new_edge_features_3, edges)
        smoothed_4          = self.layer44(node_input[:,0:2], new_node_features_4, edges, smoothing_weights)

        new_node_features_5, new_edge_features_5 = self.layer5(smoothed_4, new_edge_features_4, edges)
        smoothed_5          = self.layer55(node_input[:,0:2], new_node_features_5, edges, smoothing_weights)

        new_node_features_6, new_edge_features_6 = self.layer6(smoothed_5, new_edge_features_5, edges)
        smoothed_6          = self.layer66(node_input[:,0:2], new_node_features_6, edges, smoothing_weights)

        new_node_features_7, new_edge_features_7 = self.layer7(smoothed_6, new_edge_features_6, edges)
        smoothed_7          = self.layer77(node_input[:,0:2], new_node_features_7, edges, smoothing_weights)
		
		## output dense layer, without nonlinear activation
        node_outputs                             = self.layer8(smoothed_7[:,0:])

        return node_outputs