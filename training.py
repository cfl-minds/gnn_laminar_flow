# Custom imports
from network_utils import *
from training_utils import *
from log import logs

###################################################
###################################################

logs.info("Devices in use:")
cpus = tf.config.experimental.list_physical_devices('CPU')
for cpu in cpus:
    logs.info("Name: %s Type: %s", cpu.name, cpu.device_type)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    logs.info("Name: %s Type: %s \n", gpu.name, gpu.device_type)


# Set which GPU to use out of two V100 Teslas
useGPU = 0  # 1
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[useGPU], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        logs.info("%s Physical GPUs, %s Logical GPU", len(gpus), len(logical_gpus))
        tf.config.experimental.set_memory_growth(gpus[useGPU], True)
    except RuntimeError as e:
        logs.info(" Visible devices must be set before GPUs have been initialized")
        print(e)

###################################################
###################################################


if __name__ == '__main__':

    start      = time.time()

    # load data set
    nodes_set, edges_set, flow_set = load_dataset(2000, normalize=True, do_read=True, dataset_source='./dataset/dataset_toUse.txt')  # Dictionary with 2000 sample
    nodes_set_train, edges_set_train, flow_set_train, nodes_set_valid, edges_set_valid, flow_set_valid, nodes_set_test, edges_set_test, flow_set_test = split(nodes_set, edges_set, flow_set, train_ratio, valid_ratio)
    del nodes_set_test, edges_set_test, flow_set_test, nodes_set, edges_set, flow_set

    # declare a new model
    my_model   = InvariantEdgeModel(edge_feature_dims, num_filters, depth, mlp_width, initializer)

    # # warm start
    # my_model.load_weights('./best_model/best_model')

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

    logs.info('Training finished in %s seconds \n', (end-start))

    np.savetxt('best_model/training_loss.csv', training_loss, delimiter=',')
    np.savetxt('best_model/validation_loss.csv', validation_loss, delimiter=',')

    logs.info('The minimum validation loss attained is : %s', min(validation_loss))
