import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

model = load_model("ITOM_model")

model.summary()
