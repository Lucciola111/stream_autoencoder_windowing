import tensorflow as tf

# Call before start of training
# If GPU available: code is executed by specified GPU automatically


def setup_machine(cuda_device, ram=4096):
    """

    Parameters
    ----------
    cuda_device: Number of cuda device (GPU)
    ram: Memory Limit

    Returns
    -------

    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    CUDA_VISIBLE_DEVICE = cuda_device
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[CUDA_VISIBLE_DEVICE], 'GPU')
            tf.config.experimental.set_virtual_device_configuration(gpus[CUDA_VISIBLE_DEVICE], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=ram)])
        except RuntimeError as e:
            print(e)
