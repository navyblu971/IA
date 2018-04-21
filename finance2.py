import csv
import os
import cProfile
import time
import numpy
import pandas as pd
import warnings
import tensorflow as tf



def getData(y_name='AALBERTS'):
    """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
    #train_path, test_path = maybe_download()

    data = pd.read_csv("../cours.csv", sep=';', header=0, na_values=0,  dtype=str, decimal=',' )
    data_x, data_y = data, data.pop('749352')

    '''
    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)
    '''

    return data_x, data_y


##print (load_data())


def train_input_fn(features, labels, batch_size):
	"""An input function for training"""
	# Convert the inputs to a Dataset.
	dataset = tf.data.Dataset.from_tensor_slices( {"features": features,"labels": labels})
    #dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
	#dataset = tf.data.Dataset.from_tensor_slices((features, labels))
	# Shuffle, repeat, and batch the examples.
	#dataset = dataset.shuffle(1000).repeat().batch(batch_size)
	# Return the dataset.
	return dataset



def myTransform(x):
	#return x.replace('nan', '0')
	if type(x) == str:
		return float(x.replace(',','.' ))
	else:
		return x

def map_float(x):
	return float(x.replace(',','.'))




#filename = '../cours.csv'
#csv_delimiter =';'

#df = pandas.read_csv(filename, sep=csv_delimiter,na_values=0,  dtype=str, decimal=',' , skiprows=1)
#data = df.values
#print (data.shape)
#print (data[0,:])
#event = {'Valeur de test': data[Ø] , (for name in len (data[Ø]): print (')}

#y = numpy.zeros((2250,254),dtype=numpy.float64)


def getFeatures (DATA) :

			w, h = 254 , 2250;
			y = [[0 for x in range(w)] for y in range(h)]


			#print ("y" , y)
			#print ("y,:" , y[:][:] )
			#print ("type y:" , type(y) )
			#print (y[0, 1])


			#print("iloc" , DATA.iloc[1])

			line =1
			while line < 2248:
				#print("line-->" , line)
				tmp = list( map (myTransform ,DATA.iloc[line,1:] ) )
				#print ("tmp ", tmp)
	#y[line,:]= list (map(map_float, tmp))
				y[line-1][:]= tmp
				line+=1

			#print ("y = " , y[0][:])


			line = 2
			while (line < 2248):
				#on ne touche pas à la premiere colonne
				col=1
				while (col < 253):
					#test colonne exclue ici
					#try:
					#print(y[line-1][col])
					#print(y[line][col])
					if y[line-1][col] > y[line][col]:
						y[line][col] = 0
					else:
						y[line][col] = 1
		#except:
    			#print("Voici l'erreur :", exception_retournee)
		#		print("col", col)
		#		print("line" ,line)

					col+=1
				line+=1

			return y ##features

print ("step 1")
data, labels = getData()
print ("step 2")
features = getFeatures (data)
print ("step 3")
print (features[2248][:])

#train_input_fn(features, labels, 100)
Step = 0.1


x = tf.placeholder(tf.float32,shape=[1,1000])

y = x * 2

with tf.Session() as session:
    result = session.run(y, feed_dict={x: [1, 2, 3]})
    print(result)





# Build a DNN with 2 hidden layers and 10 nodes in each hidden layer.

'''
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 10 nodes each.
    hidden_units=[10, 10],
    # The model must choose between 3 classes.
    n_classes=3)


# Train the Model.
classifier.train(
    input_fn=train_input_fn(features, labels, 100), steps=Step)


'''

'''

x = data[1,1:].transform(',' , '.')
y[0,:] = x.astype(numpy.float)





print ("y",y)

i = 0
for value in data[1,:]:
	#events = dict( ('{}', {}).format( name, i))
	#events[name]

	if i == 0:

		if value == "NAN":
                        y[i,0] = 0


		if type(value)==str and value != "NAN":
                         y[i,0] = float(value.replace(',', '.'))


		if type(value)==float:
			y[i,0] = value

	if i >0:

		if value == "NAN":
                        y[i,0] = 0


		if type(value)==str and value != "NAN":
                         y[i,0] = float(value.replace(',', '.'))


		if type(value)==float:
			y[i,0] = value



	i+=1
'''

print ("y" , y[1][:])






#i = 0
#for d in data:

#	for j in (0, 30)
#	y[0] = d.split(csv_delimiter)[0]
#	y[i] = d.split(csv_delimiter)[j] = 1
#	print ("d", d)
