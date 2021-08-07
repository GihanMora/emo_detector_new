import tensorflow as tf
import tensorflow as tf
print(tf.__version__)
import os
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin")
# import transformers
import numpy as np
# print(transformers.__version__)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
import torch
print(torch.cuda.is_available())
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
#
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())