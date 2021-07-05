import tensorflow as tf
import logging
import os
import numpy as np


train_ratio = 0.8
valid_ratio = 0.1
test_ratio  = 1 - train_ratio - valid_ratio

viscosity         = 0.1
gravity           = 0.0
output_channels   = 3
weight_bc         = 1.5 ## the weight on boundary conditions in the navier-stokes residual
weight_mass       = 1.2 #


K                 = 3 ## specify the number of hidden layers
nb_filters        = [100, 1000, 50]## specify output channels of each hidden layer
if len(nb_filters) != K:
    print('Check the number of hidden layers again !')

# Set tf verbosity
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '10'
logger                             = tf.get_logger()
logger.setLevel(logging.ERROR)


# Set random seeds
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = '1'
tf.random.set_seed(1)
np.random.seed(1)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1)
session_conf.gpu_options.allow_growth = True

tf.compat.v1.set_random_seed(1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(),
                            config=session_conf)


tf.compat.v1.keras.backend.set_session(sess)



### training parameters
optimizer  = tf.keras.optimizers.Adam(learning_rate=1e-2)

num_epochs = 1000

initial_learning_rate = 0.002
decay_factor          = 0.002

batch_size            = 64

edge_feature_dims=4 * np.array([1,2,4,8,16,16,8,4])#9*np.ones(9, dtype=np.int32)#np.array([3,6,12,24,48,48,24,12,6,3])#6*np.ones(10, dtype=np.int32)
# [3**(5-abs(4-i)) for i in range(9)]#
num_filters=4 * np.array([2,4,8,16,16,8,4,2])#np.array([16,32,64,64,32,16])#np.ones(9, dtype=np.int32)#np.array([3,6,12,24,48,48,24,12,6,3])#3*np.ones(10, dtype=np.int32)
#num_filters[-1] = 3
initializer = tf.keras.initializers.GlorotNormal(seed=10000) #tf.random_normal_initializer(0,0.001,100)#tf.keras.initializers.GlorotNormal #tf.random_normal_initializer(0,0.00001,10)
