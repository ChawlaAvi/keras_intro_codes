from keras.models import Sequential,load_model # model imported
from keras.layers import Dense		# layer type
from keras.utils import to_categorical

import numpy as np
import pickle
try :
	model = load_model('model_multi.h5')
except:
	#declare the model
	model = Sequential()

	#add the layers in it(32 is the number of neurons in the next layer)
	model.add(Dense(32,activation ='relu',input_dim = 100 ))
	#here 1 is again the number of neuron in next layer
	#ADD DROPOUT BY -> model.add(Dropout(fraction_of_dropout_to_be_applied))
	#import Dropout from keras.layers
	model.add(Dense(10,activation='sigmoid'))

	model.compile(optimizer='rmsprop',  # we can have our own optimizer here
		loss = 'categorical_crossentropy',
		metrics=['accuracy'])

	#get the data and the labels
	data = np.random.random((1000,100))
	labels = np.random.randint(10,size=(1000,1))

	#convert to one hot labels
	one_hot_labels = to_categorical(labels, num_classes=10)

	with open('one_hot_labels.pickle','wb') as h:
		pickle.dump(one_hot_labels,h)


	model.fit(data,one_hot_labels,epochs = 400,batch_size=32)

	#print the weights 
	for layer in model.layers:
		print(layer.get_weights())

	#save the model
	model.save('model_multi.h5')
	#STORES IT as a HDF5 file. Hierarchical Data Format 
	# is a set of file formats 
	# designed to store and organize large amounts of data
	
	# delete the model
	# del model


	#load the model
    #model.summary() returns information of all the parameters, inputs etc in a tabular form

	#TEST YOUR MODEL

	#generate some random data

x_test = np.random.random((1000,100))
y_test = np.random.randint(10,size=(1000,1))
y_test = to_categorical(y_test, num_classes=10)

score = model.evaluate(x_test, y_test, batch_size=128)


#keras.evaluate ouputs the accuracy or loss. It doesnot give predictions about the unseen data
#keras.predict gives the predictions on the test data.

print(score)
