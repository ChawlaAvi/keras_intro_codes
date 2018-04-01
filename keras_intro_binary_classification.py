from keras.models import Sequential,load_model # model imported
from keras.layers import Dense		# layer type

import numpy as np

#declare the model
model = Sequential()

#add the layers in it(32 is the number of neurons in the next layer)
model.add(Dense(32,activation ='relu',input_dim = 100 ))
#here 1 is again the number of neuron in next layer
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='rmsprop',
	loss = 'binary_crossentropy',
	metrics=['accuracy'])

#get the data and the labels
data = np.random.random((1000,100))
labels = np.random.randint(2,size=(1000,1))

model.fit(data,labels,epochs = 100,batch_size=32)

#print the weights 
for layer in model.layers:
	print(layer.get_weights())

#save the model
model.save('model.h5')
#STORES IT as a HDF5 file. Hierarchical Data Format 
# is a set of file formats 
# designed to store and organize large amounts of data



# delete the model
# del model


#load the model

# model = load_model('model.h5')
