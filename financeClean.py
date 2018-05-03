
import csv
import os
import cProfile
import time
import numpy as np
import pandas as pd
import warnings
import tensorflow as tf
import itertools



def myTransform(x):
    #return x.replace('nan', '0')
    if type(x) == str:
        x.replace('nan', '0')
        return float(x.replace(',','.' ))
    else:
        return x




def getFeaturesAndLabel (file_name = "../cours.csv" , y_name='679540' ) :
            '''return features and y as column_number '''

            NB_COLUMNS = 10
            NB_ROWS = 5000
            data = pd.read_csv(file_name, sep=';',   dtype=str, decimal=',' , nrows=NB_ROWS, usecols= range(NB_COLUMNS))

            data.replace(np.nan, '0', regex=True)

            columns_list = pd.read_csv(file_name, nrows=0, sep=';' , usecols=range (NB_COLUMNS)).columns.values.tolist()

            #print (columns_list)

            ''' on enleve le y_name qui est utilisé comme label, ils sont deux fois dans la liste d'ou les deux removes ...'''

            labels_column_number = columns_list.index(y_name)

            #columns_list.remove('Identifiant')
            #columns_list.remove(y_name)
            #columns_list.remove(y_name+".1")

            number_of_rows = data.shape[0]
            number_of_columns = data.shape[1]
            print ("number of columns :",number_of_columns )
            print ("number_of_rows :",number_of_rows )
            #limite entre les features et les tests..
            type_size = int (data.shape[0] / 2 )

            y = np.zeros((number_of_rows, number_of_columns))
            features = np.zeros((number_of_rows, number_of_columns))
            w, h = number_of_columns , number_of_rows;
            #y = [[0 for x in range(w)] for y in range(h)]
            line =1
            col =0
            tmp = []
            while line < number_of_rows:
                #y[line][:]=  map (myTransform ,data.iloc[line,1:])
                '''
                print (data.iloc[line,1:])
                tmp  = data.iloc[line,1:]


                tmp = np.array (map (myTransform ,data.iloc[line,1:number_of_columns]))
                tmp = np.asarray(tmp)
                print ("tmp.shape" , tmp.shape )
                print (tmp)
                y = np.vstack((y, tmp))
                #print(line)
                #print("y", y)
                #print(tmp)
                '''
                col=1
                while col < number_of_columns :
                    y[line][col]=myTransform(data.iloc[line,col])
                    col += 1

                line+=1


            print ("y" , y)

            #features = y






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



            t=0
            targets = np.arange(0 , number_of_rows)
            while t < number_of_rows-1  :
                    #print (t)
                    targets[t] = y[t][labels_column_number]
                    t=t+1



            features = np.delete(features, (0), axis=0)
            targets = np.delete(targets, (0), axis=0)

            targets = targets.reshape(-1, 1)



            return features, targets, number_of_rows, number_of_columns, columns_list



features, targets, nb_rows, nb_columns,   COLUMNS_LIST = getFeaturesAndLabel()



features = np.delete(features, (0), axis=0)
targets = np.delete(targets, (0), axis=0)

#print ("features[0][:]", features[0][:])
#print ("features[1][:]" , features[1][:])


#features = np.asarray(features) #A REMETTRE

#features = np.random.randn(100, 80)

#print (format(features))


#test
#targets =  np.random.randn(nb_rows, nb_columns) + 3
#targets = targets.reshape(-1, 1)

#test datasets

#features, targets = (np.random.sample((nb_rows,nb_columns)), np.random.sample((nb_rows,1)))





def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    #dataset1 = tf.data.Dataset.from_tensor_slices(features)
    #dataset2 = tf.data.Dataset.from_tensor_slices(labels)
    #dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
    dataset1 = tf.data.Dataset.from_tensor_slices((features, labels))

    # Shuffle, repeat, and batch the examples.
    return dataset1.shuffle(1000).repeat().batch(batch_size)

print ("features shape :" , features.shape)
print ("target shape " , targets.shape)

#targets = np.random.sample((nb_rows,1))
#dataset = tf.data.Dataset.from_tensor_slices((features,targets))

targets = targets.reshape(-1, 1)
print ("target.dtype :" , targets.dtype)

#targets = np.asarray(targets2)
print ("targets" , targets)
firstLayerSize = nb_columns
#tf_features = tf.placeholder(tf.float32, shape=[None, nb_columns])
tf_features = tf.placeholder(tf.float32, shape=[None, firstLayerSize])
tf_targets = tf.placeholder(tf.float32, shape=[None, 1])


mydataset = train_input_fn(features, targets ,1000)
print (mydataset)

# Instantiate an estimator, passing the feature columns.
estimator = tf.estimator.LinearClassifier(
    feature_columns=COLUMNS_LIST,
    )

estimator.train(input_fn=train_input_fn(features, targets, 100), steps=2000)
