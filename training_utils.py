import math
from random import shuffle

from data_utils import *
from params import *
from log import logs


def count_neighbour_edges(node_features, edges):
    """
    return the degree of the nodes
    """
    n_nodes    = tf.shape(node_features)[0]
    n_edges = tf.shape(edges)[0]
    ones       = tf.ones((n_edges, 1))
    count      = tf.math.add(tf.math.unsorted_segment_sum(ones, edges[:, 0], n_nodes),
                             tf.math.unsorted_segment_sum(ones, edges[:, 1], n_nodes))
    return count


def mean_absolute_error(x):
    return tf.math.reduce_mean(tf.math.reduce_sum(tf.math.abs(x), axis=1))


def loss_fn(prediction, real):

    return mean_absolute_error(tf.math.subtract(prediction, real))


def watch_loss(model, nodes_set, edges_set, flow_set, do_batch=False, size_batch=64):
    """
    return the loss value of a mini-batch
    """

    shape_list = list(nodes_set.keys())
    num_graphs = len(shape_list)
    # Whether to batch or not
    if not do_batch:
        size_batch = num_graphs

    num_batch = math.ceil(num_graphs / size_batch)

    loss_of_epoch = 0.0
    total_nodes = 0.0
    for index in np.arange(num_batch):
        nodes_batch, edges_batch, flow_batch = get_batch_from_training_set(index, nodes_set, edges_set, flow_set,
                                                                           shape_list, size_batch)

        count = count_neighbour_edges(nodes_batch, edges_batch)  # determine the degree of every node
        edge_features = tf.math.reduce_mean(tf.gather(nodes_batch[:, :3], edges_batch), 1)
        outputs = model(nodes_batch[:, :3], edges_batch, edge_features, count)
        loss = loss_fn(outputs, flow_batch)

        n_nodes_batch = tf.cast(tf.shape(nodes_batch), tf.float32)[0]
        loss_of_epoch += tf.math.multiply(loss, n_nodes_batch)  # Multiply loss by number of nodes in each batch graph
        total_nodes += n_nodes_batch

    return loss_of_epoch / total_nodes


def train_model(model, nodes_set, edges_set, flow_set, optim, shape_list, num_batch, size_batch):
    loss_of_epoch = 0.0
    total_nodes = 0.0
    for index in np.arange(num_batch):
        nodes_batch, edges_batch, flow_batch = get_batch_from_training_set(index, nodes_set, edges_set, flow_set,
                                                                           shape_list, size_batch)

        count = count_neighbour_edges(nodes_batch, edges_batch)  # degree of every node = number of neighboring nodes
        edge_features = tf.math.reduce_mean(tf.gather(nodes_batch[:, :3], edges_batch), 1)  # Compute the edge features

        with tf.GradientTape() as tape:
            outputs = model(nodes_batch, edges_batch, edge_features, count)
            loss_batch = loss_fn(outputs, flow_batch)

        grads = tape.gradient(loss_batch, model.trainable_weights)
        optim.apply_gradients(zip(grads, model.trainable_weights))
        n_nodes_batch = tf.cast(tf.shape(nodes_batch), tf.float32)[0]
        loss_of_epoch += tf.math.multiply(loss_batch, n_nodes_batch) # Multiply loss by number of nodes in each batch graph
        total_nodes += n_nodes_batch

    return loss_of_epoch / total_nodes  # Divided by the total number of nodes from all batch graphs


def training_loop(model, epochs_num, nodes_set_train, edges_set_train, flow_set_train, nodes_set_valid, edges_set_valid,
                  flow_set_valid, optim, initial_rate, decayfactor, size_batch):
    training_loss   = list()
    validation_loss = [1000000000000000000.0]
   
    shape_list = list(nodes_set_train.keys())
    num_graphs = len(shape_list)
    num_batch = math.ceil(num_graphs/size_batch)  # divide the data set into num_batch mini-batches

    early_stop = 0
    for epoch in range(epochs_num):
        logs.info('Started epoch %s', epoch)
        shuffle.shuffle(shape_list)
        start      = time.time()
        # apply learning rate decay after each epoch
        learning_rate = initial_rate / (1.0 + decayfactor * epoch)
        optim.lr.assign(learning_rate)

        train_loss = train_model(model, nodes_set_train, edges_set_train, flow_set_train, optim,
                                 shape_list, num_batch, size_batch)
        
        training_loss.append(train_loss)
        ##
        valid_loss = watch_loss(model, nodes_set_valid, edges_set_valid, flow_set_valid)
        validation_loss.append(valid_loss)

        if epoch == 0:
            model.summary()
            logs.info('The model have %s learnable parameters ! \n', count_params(model.trainable_weights))

        end      = time.time()

        logs.info('Epoch %s: %s seconds --- training loss is %s --- validation loss is %s; \n', epoch, (end-start),
                  (train_loss.numpy()), (valid_loss.numpy()))

        if valid_loss < min(validation_loss[:-1]):
            early_stop = 0
            model.save_weights('./best_model/best_model_e{}'.format(epoch))
        else:
            early_stop += 1
        if early_stop == 60:
            break
    return training_loss, validation_loss[1:]
