import csv
import os
import cProfile
import time
import numpy
import pandas
import warnings


def myTransform(x ):
	float (x.replace(',', '.'))
	return



filename = '../cours.csv'
csv_delimiter =';'

df = pandas.read_csv(filename, sep=csv_delimiter, dtype=str, decimal=',' , skiprows=1)
data = df.values
print (data.shape)
print (data[0,:])
#event = {'Valeur de test': data[Ø] , (for name in len (data[Ø]): print (')}

y = numpy.zeros((2250,254),dtype=numpy.float64)

print ("y" , y)
print ("y,:" , y[0,:])


line =0
while line < 254:
	'''
	for d in data[line,:]:
	y[line,] += d
	'''
	#str = data[line,1:]
	data[line,1:] = map (myTransform ,data[line,1:] )
	y[line,:]=data[line,1:]

line+=1



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

print ("y" , y)

def myTransform(x ):
	float (x.transform (',', '.'))
	return



#i = 0
#for d in data:

#	for j in (0, 30)
#	y[0] = d.split(csv_delimiter)[0]
#	y[i] = d.split(csv_delimiter)[j] = 1
#	print ("d", d)
