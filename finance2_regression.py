import csv
import os
import cProfile
import time
import numpy as np
import pandas as pd
import warnings
import tensorflow as tf
import itertools



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
    #dataset = tf.data.Dataset.from_tensor_slices( {"features": features,"labels": labels})
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
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

            NB_COLUMNS = 10
            data = pd.read_csv(file_name, sep=';',   dtype=str, decimal=',' , nrows=100, usecols= range(NB_COLUMNS))



            columns_list = pd.read_csv(file_name, nrows=0, sep=';' , usecols=range (NB_COLUMNS)).columns.values.tolist()

            #print (columns_list)

            ''' on enleve le y_name qui est utilisé comme label, ils sont deux fois dans la liste d'ou les deux removes ...'''

            labels_column_number = columns_list.index(y_name)

            #columns_list.remove('Identifiant')
            #columns_list.remove(y_name)
            #columns_list.remove(y_name+".1")

            number_of_rows = data.shape[0]
            number_of_columns = data.shape[1]
            print ("number of columns :",data.shape[1] )
            print ("number_of_rows :",data.shape[0] )
            #limite entre les features et les tests..
            type_size = int (data.shape[0] / 2 )
            w, h = number_of_columns , number_of_rows;
            y = [[0 for x in range(w)] for y in range(h)]
            line =1
            while line < number_of_rows:
                tmp = list( map (myTransform ,data.iloc[line,1:]))
                y[line-1][:]= tmp
                #print(tmp)
                line+=1





            '''on garde la premiere ligne de donnée tel quel ..on commence à la ligne 2'''
            line = 1
            while (line < number_of_rows-1):
                #on ne touche pas à la premiere colonne
                col=0
                while (col < number_of_columns -1):
                    #test colonne exclue ici
                    #try:
                    #print(y[line-1][col])
                    #print(y[line][col])
                    #print(line)
                    #print (col)
                    if y[line-1][col] > y[line][col]:
                        y[line][col] = 0
                    else:
                        y[line][col] = 1

                    '''
                    if line <  type_size:
                        #features.append( dict(zip(columns_list,y[line])) )
                        features[columns_list[col]]+=
                    else:
                        #test_x.append( dict(zip(columns_list,y[line])) )

                    '''


                    #print (col)
                    backspace(len(str(line)))
                    #print (features)

        #except:
                #print("Voici l'erreur :", exception_retournee)
        #        print("col", col)
        #        print("line" ,line)

                    col+=1
                line+=1

            #for c in range (number_of_columns) :
            #    features[columns_list[c]] = np.asarray(y[:][c] , dtype=np.float32)



            #print ("y[:][2]" , y[:][2])

            #print ( "format feature " , features[columns_list[2]] )

            #_y = [labels_column_number][:]

            #_y =  np.asarray(y[:][labels_column_number]  , dtype=np.float32)
            #print ("format _y " , format (_y) )

            #print (features[columns_list[2]])


            #print(dict(features))

            #print ("_y :"  , _y)

            #print ("feature shape" , features[columns_list[4]].shape)
            #print ("_y shape" , _y.shape)

            #return y, y[:][column_number],columns_list ##features


            #labels = features[column_number]
            #del features[column_number]

            #print ("step 1")
            #print (labels.shape())
            #_y = dict(itertools.islice(labels.items(), number_of_rows-1))
            #print ("step 2")
            #print (_y)
            #test_y = dict(itertools.islice(labels.items(), type_size ,number_of_rows-1 ))
            #print(test_y)

            features = y
            return features, y[:][labels_column_number] , number_of_rows, number_of_columns, columns_list



features, targets, nb_rows, nb_columns,   COLUMNS_LIST = getFeaturesAndLabel()

print ("features[0][:]", features[0][:])
print ("features[1][:]" , features[1][:])



tf_features = tf.placeholder(tf.float32, shape=[None, nb_columns])
tf_targets = tf.placeholder(tf.float32, shape=[None, nb_rows])

    # First
w1 = tf.Variable(tf.random_normal([nb_columns, 3]))
b1 = tf.Variable(tf.zeros([nb_columns]))
    # Operations
z1 = tf.matmul(tf_features, w1) + b1
a1 = tf.nn.sigmoid(z1)

    # Output neuron
w2 = tf.Variable(tf.random_normal([3, 1]))
b2 = tf.Variable(tf.zeros([1]))
# Operations
z2 = tf.matmul(a1, w2) + b2
py = tf.nn.sigmoid(z2)

cost = tf.reduce_mean(tf.square(py - tf_targets))

correct_prediction = tf.equal(tf.round(py), tf_targets)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for e in range(10000):

        sess.run(train, feed_dict={
            tf_features: features,
            tf_targets: targets
        })

        print("accuracy =", sess.run(accuracy, feed_dict={
            tf_features: features,
            tf_targets: targets
        }))

#tf_features = tf.placeholder(tf.float32, shape=[None, 2])
#tf_targets = tf.placeholder(tf.float32, shape=[None, 1])
