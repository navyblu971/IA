import csv
import os
import cProfile
import time
import numpy
import pandas as pd
import warnings
import tensorflow as tf



def backspace(n):
    # print((b'\x08').decode(), end='')     # use \x08 char to go back
    print('\r', end='')                     # use '\r' to go back



def getData(y_name='749352'):
	file_name = "../cours.csv"

	#columns_list = pd.read_csv(file_name, nrows=1).columns
	data = pd.read_csv(file_name, sep=';',   dtype=str, decimal=',' , usecols=[20])
	pd.fillna(0)

	data_x, data_y = data, data.pop(y_name)

	columns_list = pd.read_csv(file_name, nrows=0, sep=';', usecols=[20]).columns.values.tolist()

	print (columns_list)

	''' on enleve le y_name qui est utilisé comme label, ils sont deux fois dans la liste d'ou les deux removes ...'''
	columns_list.remove('Identifiant')
	columns_list.remove(y_name)
	columns_list.remove(y_name+".1")

	#df = pd.read_csv(file_name, nrows=1)
	#columns_list = list(df.columns.values)
	#columns_list = list (range(255))


	'''
    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)
	'''

	return data_x, data_y, columns_list

	##print (load_data())



def _parse_line(COLUMNS_NAME,column_number ,  line):
    # Decode the line into its fields


    # Pack the result into a dictionary
    features = dict(zip(COLUMNS_NAME,line))
	#label = features[column_number]
    return features, label




def train_input_fn(features, labels, batch_size):
	"""An input function for training"""
	# Convert the inputs to a Dataset.
	dataset = tf.data.Dataset.from_tensor_slices( {"features": features,"labels": labels})
    #dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
	#dataset = tf.data.Dataset.from_tensor_slices((features, labels))
	# Shuffle, repeat, and batch the examples.
	dataset = dataset.shuffle(1000).repeat().batch(batch_size)
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


def getFeaturesAndLabel (file_name = "../cours.csv" , y_name='749352' ) :
			'''return features and y as column_number '''

			NB_COLUMNS = 20
			data = pd.read_csv(file_name, sep=';',   dtype=str, decimal=',' , nrows=1100 , usecols= range(NB_COLUMNS))



			columns_list = pd.read_csv(file_name, nrows=0, sep=';' , usecols=range (NB_COLUMNS)).columns.values.tolist()

			#print (columns_list)

			''' on enleve le y_name qui est utilisé comme label, ils sont deux fois dans la liste d'ou les deux removes ...'''

			column_number = columns_list.index(y_name)

			#columns_list.remove('Identifiant')
			#columns_list.remove(y_name)
			#columns_list.remove(y_name+".1")

			number_of_rows = data.shape[0]
			number_of_columns = data.shape[1]

			w, h = number_of_columns , number_of_rows;
			y = [[0 for x in range(w)] for y in range(h)]


			features = []
			#dict(zip(columns_list,[0 for i in columns_list]))


			print (features)



			line =1
			'''
			while line < number_of_rows:
				tmp = list( map (myTransform ,pd.iloc[line,1:] ) )
				y[line-1][:]= tmp
				line+=1
			'''

			#print ("y = " , y[0][:])

			'''on garde la premiere ligne de donnée tel quel ..on commence à la ligne 2'''
			line = 2
			while (line < number_of_rows):
				#on ne touche pas à la premiere colonne
				col=1
				while (col < number_of_columns):
					#test colonne exclue ici
					#try:
					#print(y[line-1][col])
					#print(y[line][col])
					if y[line-1][col] > y[line][col]:
						y[line][col] = 0
					else:
						y[line][col] = 1

					features.append( dict(zip(columns_list,y[line])) )
					print (line)
					backspace(len(str(line)))
					#print (features)

		#except:
    			#print("Voici l'erreur :", exception_retournee)
		#		print("col", col)
		#		print("line" ,line)

					col+=1
				line+=1


			print (features[2])
			#return y, y[:][column_number],columns_list ##features


			labels = features[column_number]
			del features[column_number]
			return features, labels, columns_list




print ("step 1")
features, labels, COLUMNS_LIST= getFeaturesAndLabel()
print ("step 2")
#features, labels = getFeaturesAndLabel (data , 3)
print ("step 3")
#print (features[2248][:])
#print (labels)
print ("step 4")

#train_input_fn(features, labels, 100)
Step = 0.1

my_feature_columns = [
    tf.feature_column.numeric_column(name)
    for name in COLUMNS_LIST]


print ("step 5")





# Build a DNN with 2 hidden layers and 10 nodes in each hidden layer.

print ("create classifier ...")
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 10 nodes each.
    hidden_units=[10, 10],
    # The model must choose between 3 classes.
    n_classes=3)






print ("train the model ..")
# Train the Model.
classifier.train(
    input_fn=train_input_fn(features, labels, 100), steps=Step)



eval_result = classifier.evaluate(
	input_fn=train_input_fn(features, labels,100),  steps=Step)


print (eval_result)
