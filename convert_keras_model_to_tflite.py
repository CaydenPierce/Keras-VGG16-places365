#https://www.tensorflow.org/lite/convert#python_api
import sys

import numpy as np
import tensorflow as tf
from keras.models import load_model

if len(sys.argv) < 2:
    print("Error. Correct usage: `python convert...py <model file location>`")
#Load the model
print(sys.argv[1])
model = load_model(sys.argv[1])

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model.
new_model_name = sys.argv[1].split(".")[0] + ".tflite"
with open(new_model_name, 'wb') as f:
  f.write(tflite_model)

# optimize the model.
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
#tflite_quant_model = converter.convert()
#
## Save the quantized converted model.
#new_model_name = sys.argv[1].split(".")[0] + "_quantized.tflite"
#with open(new_model_name, 'wb') as f:
#  f.write(tflite_quant_model)
