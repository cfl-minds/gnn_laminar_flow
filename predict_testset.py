from network_utils import *
from training_utils import *
import numpy as np
import matplotlib.pyplot as plt


# Load model
best_model   = InvariantEdgeModel(edge_feature_dims, num_filters, initializer)
best_model.load_weights('./best_model/best_model_e713')

# Load dataset
nodes_set, edges_set, flow_set = load_dataset(2000, True, do_read=True)
nodes_set_train, edges_set_train, flow_set_train, nodes_set_valid, edges_set_valid, flow_set_valid, nodes_set_test, edges_set_test, flow_set_test = split(nodes_set, edges_set, flow_set, train_ratio, valid_ratio)
del nodes_set, edges_set, flow_set #, nodes_set_train, edges_set_train, flow_set_train, nodes_set_valid, edges_set_valid, flow_set_valid

# Specify the dataset to compute the MAE
operate_on = 'Testing'  # 'Training' 'Testing' 'Validation'
if operate_on == "Validation":
    nodes_dict = nodes_set_valid
    edges_dict = edges_set_valid
    flow_dict = flow_set_valid
elif operate_on == "Testing":
    nodes_dict = nodes_set_test
    edges_dict = edges_set_test
    flow_dict = flow_set_test
else:
    nodes_dict = nodes_set_train
    edges_dict = edges_set_train
    flow_dict = flow_set_train

# Compute the MAE
shape_list = nodes_dict.keys()
MAE_testset = list()

start = time.time()
for shape in shape_list:
    nodes = nodes_dict[shape]
    edges = edges_dict[shape]
    flow  = flow_dict[shape]

    count = count_neighbour_edges(nodes, edges)
    edge_features = tf.math.reduce_mean(tf.gather(nodes[:,:3], edges), 1)
    pred = best_model(nodes[:,:3], edges, edge_features, count)
    MAE_testset.append(loss_fn(pred, flow).numpy())

# Save results of predictions
# np.savetxt('best_model/MAE_testset.csv', MAE_testset, delimiter=',', header='MAE', fmt='%1.16f', comments='')
print(np.mean(MAE_testset))
end = time.time()
print(end - start)

    # body_force = np.zeros((n, 4))
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
# np.savetxt('best_model/MSE_testset.csv', MSE_testset, delimiter=',', header='MSE', fmt='%1.16f', comments='')
