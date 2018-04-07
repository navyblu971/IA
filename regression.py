import numpy as np
import numpy.linalg as alg



f = open("train-images-idx3-ubyte", "rb")
magic_number = f.read(4)
nb_items = f.read(4)
nb_rows = f.read(4) 
nb_colums = f.read(4)

MATRICE_A = []
#MATRICE_A = np.onces((28*28,10))
MATRICE_A =np.random.rand(28*28,10) * 0.01

print MATRICE_A


def initDB ():
	f = open("train-images-idx3-ubyte", "rb")
        magic_number = f.read(4)
        nb_items = f.read(4)
        nb_rows = f.read(4) 
        nb_colums = f.read(4)
	return f



def readNext_X (f):
	

	

epoch = 1000
while epoch !=0 :
	X = readNext_X ()
	eval =  MATRICE_A.dot(X)
	norm(Y-eval)
	derivation (norm(Y-eval))
	epoch -=1

	

images =[]

i=0
'''
while i < nb_items:
	image = f.read (28*28)
	images +=image
	i+=1

print ("images", images)


g = open("train-labels-idx1-ubyte", "rb")
yMagicNumber = g.read (4)
ynb_items = g.read (4)

y = []
i=0
while i < ynb_items:
	y+=g.read(1)
	i+=1

print ("y", y)



print("magic number", magic_number)
print("number of images", nb_items)
print("nb_rows" , nb_rows)
print("nb_colums",nb_colums)


f.close()

g.close()
'''

