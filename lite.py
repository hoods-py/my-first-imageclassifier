import tensorflow as tf
import numpy as np
from PIL import Image
# from main import class_names

# The path to the saved TensorFlow Lite model
TF_MODEL_FILE_PATH = 'model.tflite'

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH)
interpreter.allocate_tensors()

class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
# Get the input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Assume the input shape is (1, height, width, channels)
input_shape = input_details[0]['shape']

# Load a sunflower image (replace 'sunflower.jpg' with your image file)
sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)
sunflower_img = Image.open(sunflower_path).resize((180, 180))

# Preprocess the image
input_data = np.array(sunflower_img, dtype=np.float32)
input_data /= 255.0  # Normalize pixel values to [0, 1]
input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension

# Set the input tensor to the loaded data
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run the inference
interpreter.invoke()

# Get the output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])

# Assuming output_data contains the predictions
# Print or use the predictions as needed
print(output_data)

# Apply softmax to get probabilities
probabilities = tf.nn.softmax(output_data)

# Get the predicted class index
predicted_class_index = np.argmax(probabilities)

# Get the predicted class name from class_names
predicted_class_name = class_names[predicted_class_index]

# Print the results
print("Predicted Class Index:", predicted_class_index)
print("Predicted Class Name:", predicted_class_name)
print("Class Probabilities:", probabilities)