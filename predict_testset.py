import pandas as pd
import numpy as np
import tensorflow as tf
import time

from data_utils import *
from params import *
from network_utils import *
from training_utils import *
from loss import MSE

best_model   = invariant_edge_model(edge_feature_dims, num_filters, initializer)
best_model.load_weights('./best_model/best_model')


nodes_set, edges_set, flow_set = load_dataset(2000, True)


nodes_set_train, edges_set_train, flow_set_train, nodes_set_valid, edges_set_valid, flow_set_valid, nodes_set_test, edges_set_test, flow_set_test = split(nodes_set, edges_set, flow_set, train_ratio, valid_ratio)


nodes_set_test, edges_set_test, flow_set_test
del nodes_set, edges_set, flow_set, nodes_set_train, edges_set_train, flow_set_train, nodes_set_valid, edges_set_valid, flow_set_valid


min_values = tf.constant([-0.13420522212982178, -0.830278217792511, -1.9049606323242188], dtype=tf.dtypes.float32, shape=(3,))
max_values = tf.constant([1.4902634620666504, 0.799094557762146, 1.558414101600647], dtype=tf.dtypes.float32, shape=(3,))


shape_list = nodes_set_test.keys()
n = len(shape_list)
MAE_testset = list()
MSE_testset = list()
body_force = np.zeros((n,4))

start = time.time()
i = 0
for shape in shape_list:
    nodes = nodes_set_test[shape]
    edges = edges_set_test[shape]
    flow  = flow_set_test[shape]

    count = count_neighb_elements(nodes, edges)
    edge_features = tf.math.reduce_mean(tf.gather(nodes[:,:3], edges), 1)
    pred = best_model(nodes[:,:3], edges, edge_features, count)
    MAE_testset.append(loss_fn(pred, flow).numpy())

    #elements= pd.read_csv('../data/elements/' + shape).values
    #_, elements = np.unique(elements, return_inverse=True)
    #elements = np.reshape(elements, (-1,3))


    #flow = tf.math.add(tf.math.multiply(flow, tf.math.subtract(max_values, min_values)), min_values)
    #pred = tf.math.add(tf.math.multiply(pred, tf.math.subtract(max_values, min_values)), min_values)
    #D1, L1 = lift_drag(nodes.numpy(), edges.numpy(), elements, flow.numpy(), 0.1)
    #D2, L2 = lift_drag(nodes.numpy(), edges.numpy(), elements, pred.numpy(), 0.1)
    #body_force[i, :] = [L1, D1, L2, D2]
    #i += 1

#np.savetxt('best_model/body_force.csv', body_force, delimiter=',', header='lift,drag,lift_pred,drag_pred', fmt='%1.16f', comments='')
np.savetxt('best_model/MAE_testset.csv', MAE_testset, delimiter=',', header='MAE', fmt='%1.16f', comments='')
np.savetxt('best_model/MSE_testset.csv', MSE_testset, delimiter=',', header='MSE', fmt='%1.16f', comments='')
print(np.mean(MAE_testset))
end = time.time()
print(end - start)
