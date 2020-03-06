from keras import models
from keras import layers

data_count = 10000
input_dims = 32
attention_column = 7
time_steps = 20

inputs = layers.Input(shape=(time_steps, input_dims,))
layer = layers.LSTM(input_dims, return_sequences=True)(inputs)
layer = layers.Permute((2, 1))(layer)
layer = layers.Dense(time_steps, activation='softmax')(layer)
layer_prob = layers.Permute((2, 1), name='attention_vec')(layer)
attention_mul = layers.multiply([inputs, layer_prob])
attention_mul = layers.Flatten()(attention_mul)
outputs = layers.Dense(64)(attention_mul)
outputs = layers.Dense(1, activation='softmax')(outputs)
model = models.Model(inputs, outputs)

model.summary()