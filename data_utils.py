import tensorflow as tf
from scipy.sparse import diags
from scipy.interpolate import griddata
import pandas as pd
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import os
import time
import progress.bar


def lift_drag(nodes, edges, elements, flow, viscosity):
    start = time.time()
    nodes[:,0:2] = 4.0 * nodes[:,0:2] - 2.0

    #### get the index and coordinates of nodes on the obstacle
    obstacle     = np.where(nodes[:,2]==0.0)[0]
    coord_obstacle = nodes[obstacle,:2]
    min_index    = np.argmin(coord_obstacle[:,0])
    node_left    = obstacle[min_index]
    ## edges associated to the obstacle
    edges_obstacle = np.asarray([edges[i,0] in obstacle for i in range(edges.shape[0])]) * np.asarray([edges[i,1] in obstacle for i in range(edges.shape[0])])
    edges_obstacle = edges[edges_obstacle,:]

    #### sort these nodes in innverse clockwise order
    sorted_nodes   = [node_left]
    while len(sorted_nodes) < edges_obstacle.shape[0]:
        three_nodes = np.unique(edges_obstacle[np.logical_or(edges_obstacle[:,0]==sorted_nodes[-1], edges_obstacle[:,1]==sorted_nodes[-1])])
        coord_three_nodes = [nodes[i,:2] for i in three_nodes]
        neighbours = np.setdiff1d(three_nodes, sorted_nodes)
        if len(neighbours) == 1:
            sorted_nodes.append(neighbours[0])
        else:
            k0 = (nodes[neighbours[0],1] - nodes[sorted_nodes[-1],1])
            k1 = (nodes[neighbours[1],1] - nodes[sorted_nodes[-1],1])
            unclockwise = np.asarray(neighbours)[~np.asarray([k0>0, k1>0])][0]
            sorted_nodes.append(unclockwise)
    #print(sorted_nodes)
    sorted_edges = np.zeros((len(sorted_nodes),2), 'int32')
    for i in range(len(sorted_nodes)-1):
        sorted_edges[i,:] = sorted_nodes[i:i+2]
    sorted_edges[-1,:] = [sorted_nodes[-1], sorted_nodes[0]]
    ### calculate and aggregate edge-wise bidy forces
    drag_lift = 0#np.zeros(2)
    
    arclen = 0
    for edge in sorted_edges:
        
        node0 = nodes[edge[0], :2]
        u0    = flow[edge[0], 0]
        v0    = flow[edge[0], 1]
        p0    = flow[edge[0], 2]
        node1 = nodes[edge[1], :2]
        u1    = flow[edge[1], 0]
        v1    = flow[edge[1], 1]
        p1    = flow[edge[1], 2]

        tangent = node1 - node0
        normal  = np.asarray([-tangent[1], tangent[0]])
		
        ##calculate pressure force
        force_p   = 0.5 * (p0 + p1) * normal
        arclen += np.linalg.norm(tangent)

        associated_element = elements[[len(np.setdiff1d(edge, elements[i,:]))==0 for i in range(elements.shape[0])]][0]
        associated_node    = np.setdiff1d(associated_element, edge)[0]
        ### turn inverse clockwise edge diretion into inverse upwind
        variable = np.dot(np.asarray(flow[associated_node,:2]), tangent)
        if variable < 0.0:
            tangent = -tangent
            variable = -variable
        length  = np.linalg.norm(tangent)
            
        local_matrix = np.hstack([np.asarray([nodes[i,:2] for i in associated_element]), np.ones((3,1))])
        force_nu  = viscosity * tangent * variable / np.linalg.det(local_matrix)
        
        drag_lift = drag_lift + force_nu + force_p

    end = time.time()
    print(drag_lift)
    print(end - start)
    return drag_lift





def split(nodes_set, edges_set, flow_set, train_ratio, valid_ratio):

    shape_list  = list(nodes_set.keys())
    n           = len(shape_list)
    
    index_train = int(n*train_ratio)
    index_valid = int(n*(train_ratio+valid_ratio))
    
    keys_train  = shape_list[0:index_train]
    keys_valid  = shape_list[index_train:index_valid]
    keys_test   = shape_list[index_valid:]
    nodes_set_train  = {key: nodes_set[key] for key in keys_train}
    nodes_set_valid  = {key: nodes_set[key] for key in keys_valid}
    nodes_set_test   = {key: nodes_set[key] for key in keys_test}

    edges_set_train  = {key: edges_set[key] for key in keys_train}
    edges_set_valid  = {key: edges_set[key] for key in keys_valid}
    edges_set_test   = {key: edges_set[key] for key in keys_test}

    flow_set_train   = {key: flow_set[key] for key in keys_train}
    flow_set_valid   = {key: flow_set[key] for key in keys_valid}
    flow_set_test    = {key: flow_set[key] for key in keys_test}
    

    print('Data set is split is split to {} training set, {} validation set and {} test set !'.format(len(keys_train), len(keys_valid), len(keys_test)))
    print(' ')
    return nodes_set_train, edges_set_train, flow_set_train, nodes_set_valid, edges_set_valid, flow_set_valid, nodes_set_test, edges_set_test, flow_set_test




def load_data(file_name, normalize):

    nodes = pd.read_csv('../data/nodes/'+file_name)[['x', 'y', 'Object']].values.astype('float32')#, 'u_bc1', 'u_bc2', 'dist0']]
    flow  = pd.read_csv('../data/flow/'+file_name).values.astype('float32')
    edges = pd.read_csv('../data/edges/'+file_name).values
    nodes = nodes[np.unique(edges),:]
    flow = flow[np.unique(edges),:]

    ##  reset node index
    _, edges = np.unique(edges, return_inverse=True)
    edges = np.reshape(edges, (-1,2))


    flow  = tf.convert_to_tensor(flow, dtype=tf.dtypes.float32)
    return tf.convert_to_tensor(nodes, dtype=tf.dtypes.float32), tf.convert_to_tensor(edges, dtype=tf.dtypes.int32), flow, tf.math.reduce_max(flow, axis=0), tf.math.reduce_min(flow, axis=0)


def load_dataset(num_files, normalize):
    """
    Load num_files data from the data direction.
    : normalize: apply a channel wise normaliztion to the flow field (u, v, p)
    """
    nodes_set       = dict()
    edges_set       = dict()
    flow_set     = dict()

    file_list = os.listdir('../data/nodes/')[:num_files]
    print('The data set contains {} instances !'.format(len(file_list)))
    print(' ')

    max_value = tf.zeros((3,), dtype=tf.dtypes.float32)
    min_value = tf.zeros((3,), dtype=tf.dtypes.float32)
    
    
    ### 
    bar = progress.bar.Bar('Loading data set ' , max=num_files)
    for file_name in file_list:

        nodes, edges, flow, max_uvp, min_uvp   = load_data(file_name, normalize)

        nodes_set[file_name]        = nodes
        edges_set[file_name]       = edges
        flow_set[file_name]     = flow

        max_value            = tf.math.reduce_max([max_value, max_uvp], axis=0)
        min_value            = tf.math.reduce_min([min_value, min_uvp], axis=0)
        bar.next()
    bar.finish()

    ### channel-wise normalization, mapping u/v/p values into [0, 1]
    if normalize == True:
        for file_name in file_list:

            x = tf.math.divide(tf.math.subtract(nodes_set[file_name][:,0:1], -2.0), 4.0)
            y = tf.math.divide(tf.math.subtract(nodes_set[file_name][:,1:2], -2.0), 4.0)
            objet    = tf.reshape(nodes_set[file_name][:,-1], (-1,1))
            nodes_set[file_name] = tf.concat([x,y,objet], axis=1)
            flow_set[file_name]  = tf.math.divide(tf.math.subtract(flow_set[file_name], min_value), tf.math.subtract(max_value, min_value))
        print('(u, v, p) value range was [{}, {}]x[{}, {}]x[{}, {}]'.format(min_value[0], max_value[0], min_value[1], max_value[1], min_value[2], max_value[2]), ' !\n', 'Now it is mapped into [0,1]x[0,1]x[0,1] !')
    else:
        print('Normalization is not applied !')
    
    print(' ')
    print('The dataset is loaded !')
    print(' ')

    return nodes_set, edges_set, flow_set



def count_params(trainable_weights):
    
    """
    Count the number of trainable parameters in a model
    """
    
    count = 0
    for weight in trainable_weights:
        try:
            count += tf.shape(weight).numpy()[0] * tf.shape(weight).numpy()[1]# * tf.shape(weight).numpy()[2]
        except:
            count += tf.shape(weight).numpy()[0]
    return count



def prepare_graph_batch(nodes_set, edges_set, flow_set):
    
    """
    Construct a big graph from a list of disjoint small graphs. 
    The big graph allows mini-batch training, and supervising loss on validation set 
    : nodes_set: a dictionary, key is the graph name, value is the node coordinates
    : edges_set: a dictionary, key is the graph name, value is the connectivity matrix
    : flow_set : a dictionary, key is the graph name, value is the (u,v,p) matrix
    """
    start       = time.time()
    shape_list  = list(nodes_set.keys())
    num_graphs = len(shape_list)

    ## Begin concatenating the coordinates array, the connectivity array and the flow array
    all_nodes    = nodes_set[shape_list[0]]
    all_edges    = edges_set[shape_list[0]]
    all_flow     = flow_set[shape_list[0]]

    for i in np.arange(num_graphs-1) + 1:

        shape    = shape_list[i]

        nodes    = nodes_set[shape]
        edges    = tf.math.add(edges_set[shape], int(tf.shape(all_nodes)[0]))
        flow     = flow_set[shape]
        
        all_nodes    = tf.concat([all_nodes, nodes], axis=0)
        all_edges    = tf.concat([all_edges, edges], axis=0)
        all_flow     = tf.concat([all_flow, flow], axis=0)

    end = time.time()
    # print('A graph with {} nodes and {} edges has been constructed in {} seconds!'.format(int(tf.shape(all_nodes)[0]), int(tf.shape(all_edges)[0]), end - start))



    return all_nodes, all_edges, all_flow

def get_batch_from_training_set(batch_index, nodes_set, edges_set, flow_set, shape_list, batch_size):

    """
    Fetch a number of graphs from the training set, return a big graph.
    : batch_index: a iterable integer, it must not be larger than len(nodes_set)/batch_size
    : nodes_set  : a dictionary, key is the graph name, value is the node coordinates
    : edges_set  : a dictionary, key is the graph name, value is the connectivity array
    : flow_set   : a dictionary, key is the graph name, value is the (u,v,p) matrix
    : batch_size : number of graphs in a mini batch, an interger smaller than len(nodes_set)
    """
     
    try:
        keys_batch  = shape_list[batch_index*batch_size: (batch_index+1)*batch_size]
    except:
        keys_batch  = shape_list[batch_index*batch_size: ]
    
    nodes_set_batch  = {key: nodes_set[key] for key in keys_batch}
    edges_set_batch  = {key: edges_set[key] for key in keys_batch}
    flow_set_batch   = {key: flow_set[key] for key in keys_batch}

    nodes_batch, edges_batch, flow_batch = prepare_graph_batch(nodes_set_batch, edges_set_batch, flow_set_batch)

    return nodes_batch, edges_batch, flow_batch


