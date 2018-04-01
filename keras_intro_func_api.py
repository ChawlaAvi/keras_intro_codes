from keras.models import Model
from keras.layers import Input,Dense

inputs = Input((784,))

x = Dense(64,activation="relu")(inputs)

pred = Dense(10, activation="softmax")(x)

model = Model(inputs=inputs, outputs = pred)
model.compile(optimizer = 'rmsprop',
	loss = 'categorical_crossentropy',
	metrices = ['accuracy'])
