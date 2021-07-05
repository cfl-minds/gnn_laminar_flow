import tensorflow as tf
import time
import math
from random import shuffle

from data_utils import *
from params import *

def count_neighb_edges(node_features, edges):
    """
    return the degree of the nodes
    """
    n_nodes    = tf.shape(node_features)[0]
    n_edges = tf.shape(edges)[0]
    ones       = tf.ones((n_edges,1))
    count      = tf.math.add(tf.math.unsorted_segment_sum(ones, edges[:,0], n_nodes), tf.math.unsorted_segment_sum(ones, edges[:,1], n_nodes))
    return count

def MAE(x):
    return tf.math.reduce_mean(tf.math.abs(x))
	
def loss_fn(pred, real):

    return MAE(tf.math.subtract(pred, real))

def watch_loss(model, nodes_set, edges_set, flow_set):
    """
    return the loss value of a mini-batch
    """
    all_nodes, all_edges, all_flow = prepare_graph_batch(nodes_set, edges_set, flow_set)
    count               = count_neighb_edges(all_nodes, all_edges)

    edge_features = tf.math.reduce_mean(tf.gather(all_nodes[:,:3], all_edges), 1) 
    outputs = model(all_nodes[:,:3], all_edges, edge_features, count)#all_nodes[:,2:6]
    loss    = loss_fn(outputs, all_flow)

    return float(loss)


def train_model(model, nodes_set, edges_set, flow_set, optimizer, learning_rate, shape_list, num_batch, batch_size):

    loss_of_epoch = 0.0
    for index in np.arange(num_batch):

        nodes_batch, edges_batch, flow_batch = get_batch_from_training_set(index, nodes_set, edges_set, flow_set, shape_list, batch_size)
        count = count_neighb_edges(nodes_batch, edges_batch)
        edge_features = tf.math.reduce_mean(tf.gather(nodes_batch[:,:3], edges_batch), 1)
		
        with tf.GradientTape() as tape:
            outputs = model(nodes_batch, edges_batch, edge_features, count)
            loss_batch = loss_fn(outputs, flow_batch)

        grads = tape.gradient(loss_batch, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        loss_of_epoch += float(loss_batch)

    return loss_of_epoch / num_batch

def training_loop(model, num_epochs, nodes_set_train, edges_set_train, flow_set_train, nodes_set_valid, edges_set_valid, flow_set_valid, optimizer, initial_learning_rate, decay_factor, batch_size):

    training_loss   = list()
    validation_loss = [1000000000000000000.0]
   
    shape_list = list(nodes_set_train.keys())
    #shuffle(shape_list)
    num_graphs = len(shape_list)
    num_batch = math.ceil(num_graphs/batch_size) ## divide the data set into num_batch mini-batches

    early_stop = 0
    for epoch in range(num_epochs):
        shuffle(shape_list)
        start      = time.time()
        ##apply learning rate decay after each epoch
        learning_rate = initial_learning_rate / (1.0 + decay_factor * epoch)
        optimizer.lr.assign(learning_rate)
        train_loss = train_model(model, nodes_set_train, edges_set_train, flow_set_train, optimizer, learning_rate, shape_list, num_batch, batch_size)
        
        training_loss.append(train_loss)
        ##
        valid_loss = watch_loss(model, nodes_set_valid, edges_set_valid, flow_set_valid)#, False)
        validation_loss.append(valid_loss)

        if epoch == 0:
            model.summary()
            print('The model have {} learnable parameters ! '.format(count_params(model.trainable_weights)))
            print(' ')
        end      = time.time()

        print('Epoch {}: {} seconds --- training loss is {} --- validation loss is {};'.format(epoch, end-start, train_loss, valid_loss))
        print(' ')

        if valid_loss < min(validation_loss[:-1]):
            early_stop = 0
            model.save_weights('./best_model/best_model')
        else:
            early_stop += 1
        if early_stop == 60:
            break
    return training_loss, validation_loss[1:]
