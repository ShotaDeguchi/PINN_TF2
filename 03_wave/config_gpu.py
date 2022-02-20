"""
********************************************************************************
configures gpu
********************************************************************************
"""

import tensorflow as tf
from tensorflow.python.client import device_lib

# gpu_flg = 1   # 0, 1, or 2 (see: https://www.tensorflow.org/guide/gpu)
#               # 0: Restrict TensorFlow to only use the first GPU
#               # 1: Find the first GPU, and restrict the memory usage (adaptive)
#               # 2: Create 2 virtual GPUs with 1GB memory each

def config_gpu(gpu_flg):
    # gpu configuration
    tf.debugging.set_log_device_placement(False)   # True to check executing device carefully
    gpus = tf.config.experimental.list_physical_devices("GPU")
    
    if gpu_flg == 0:
        print("gpu_flg:", gpu_flg)
        if gpus:
            # Restrict TensorFlow to only use the first GPU
            try:
                tf.config.experimental.set_visible_devices(gpus[0], "GPU")
                logical_gpus = tf.config.experimental.list_logical_devices("GPU")
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
            except RuntimeError as e:
                # Visible devices must be set before GPUs have been initialized
                print(e)
               
    elif gpu_flg == 1:
        print("gpu_flg:", gpu_flg)
        if gpus:
            # Find the first GPU, and restrict the memory usage (adaptive)
            try:
                tf.config.experimental.set_visible_devices(gpus[0], "GPU")
                tf.config.experimental.set_memory_growth(gpus[0], True)
                logical_gpus = tf.config.experimental.list_logical_devices("GPU")
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
                print("\nDevice information;")
                print(device_lib.list_local_devices())
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

    elif gpu_flg == 2:
        print("gpu_flg:", gpu_flg)
        if gpus:
            # Create 2 virtual GPUs with 1GB memory each
            try:
                tf.config.experimental.set_virtual_device_configuration(
                        gpus[0],
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit = 2 ** 10),
                         tf.config.experimental.VirtualDeviceConfiguration(memory_limit = 2 ** 10)])
                logical_gpus = tf.config.experimental.list_logical_devices("GPU")
                print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
                print("\nDevice information;")
                print(device_lib.list_local_devices())
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)
    
    else:
        raise NotImplementedError(">>>>> gpu_config")

if __name__ == "__main__":
    config_gpu()