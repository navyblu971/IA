import csv
import os
import cProfile
import time
import numpy
import pandas
import warnings


def myTransform(x):
	#return x.replace('nan', '0')
	return x

def map_float(x):
	return float(x.replace(',','.'))




filename = '../cours.csv'
csv_delimiter =';'

df = pandas.read_csv(filename, sep=csv_delimiter,na_values=0,  dtype=str, decimal=',' , skiprows=1)
data = df.values
#print (data.shape)
#print (data[0,:])
#event = {'Valeur de test': data[Ø] , (for name in len (data[Ø]): print (')}

#y = numpy.zeros((2250,254),dtype=numpy.float64)
w, h = 254 , 2250;
y = [[0 for x in range(w)] for y in range(h)]


print ("y" , y)
print ("y,:" , y[:][:] )


line =0
while line < 254:
	tmp = list( map (myTransform ,data[line,1:] ) )
	print (tmp)
	#y[line,:]= list (map(map_float, tmp))
	y[line][:]= tmp
	line+=1


line = 0
while (line < 254):
	#on ne touche pas à la premiere colonne
	col=0
	while (col < 2250):
		#test colonne exclue ici
		if y[line-1][col] > y[line][col]:
			y[line][col] = 0
		else:
			y[line][col] = 1

		col+=1
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






#i = 0
#for d in data:

#	for j in (0, 30)
#	y[0] = d.split(csv_delimiter)[0]
#	y[i] = d.split(csv_delimiter)[j] = 1
#	print ("d", d)
