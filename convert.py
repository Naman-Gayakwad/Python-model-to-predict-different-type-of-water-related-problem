import tensorflow as tf
from keras.models import load_model, save_model

# Load the .p model
model = load_model('model.p')

# Save the model as an h5 file
save_model(model, 'model.h5')