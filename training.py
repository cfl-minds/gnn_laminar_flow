# general imports
import logging
import os
import time
import numpy as np
import tensorflow as tf

# custom imports
from params import *
from network_utils import *
from data_utils import *
from training_utils import *

###############################################################
###############################################################
###############################################################
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

print("Devices in use:")
cpus = tf.config.experimental.list_physical_devices('CPU')
for cpu in cpus:
    print("Name:", cpu.name, "  Type:", cpu.device_type)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print("Name:", gpu.name, "  Type:", gpu.device_type)
print('')

## Set which GPU to use out of two V100 Teslas
GPU_to_use = 0

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first or second GPU
    useGPU = GPU_to_use
    try:
        tf.config.experimental.set_visible_devices(gpus[useGPU], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        tf.config.experimental.set_memory_growth(gpus[useGPU], True)
    except RuntimeError as e:
        print(" Visible devices must be set before GPUs have been initialized")
        print(e)
###############################################################
###############################################################
###############################################################		

start      = time.time()
### load data set
nodes_set, edges_set, flow_set = load_dataset(2000, True)
nodes_set_train, edges_set_train, flow_set_train, nodes_set_valid, edges_set_valid, flow_set_valid, nodes_set_test, edges_set_test, flow_set_test = split(nodes_set, edges_set, flow_set, train_ratio, valid_ratio)
del nodes_set_test, edges_set_test, flow_set_test, nodes_set, edges_set, flow_set


### declare a new model
my_model   = invariant_edge_model(edge_feature_dims, num_filters, initializer)

## warm start
#my_model.load_weights('./best_model/best_model')

training_loss, validation_loss = training_loop(my_model,
                                               num_epochs,
                                               nodes_set_train,
                                               edges_set_train,
                                               flow_set_train,
                                               nodes_set_valid,
                                               edges_set_valid,
                                               flow_set_valid,
                                               optimizer,
                                               initial_learning_rate,
                                               decay_factor,
                                               batch_size)

end        = time.time()

print('Training finished in {} seconds.'.format(end-start))
print(' ')

np.savetxt('best_model/training_loss.csv', training_loss, delimiter=',')
np.savetxt('best_model/validation_loss.csv', validation_loss, delimiter=',')

print('The minimum validation loss attained is : ' + str(min(validation_loss)) + '.')
