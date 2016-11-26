import struct
from pylab import *
from numpy import *
from array import array as pyarray
import numpy as np

#loading and reading data from the mnist data set

def read_data(digits=np.arange(10)):
	imageFile =  './train-images.idx3-ubyte'
        labelFile =  './train-labels.idx1-ubyte'
        imageFile1 = './t10k-images.idx3-ubyte'
        labelFile1 = './t10k-labels.idx1-ubyte'
	filelabel = open(labelFile, 'rb')
	magic_nr, size = struct.unpack(">II", filelabel.read(8))
	labelArr = pyarray("b", filelabel.read())
	filelabel.close()

	fimg = open(imageFile, 'rb')
	magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
	img = pyarray("B", fimg.read())
	fimg.close()

	ind = [ k for k in range(size) if labelArr[k] in digits ]
	N = len(ind)
	images = zeros((N, rows, cols), dtype=uint8)
	labels = zeros((N, 1), dtype=int8)
	for i in range(len(ind)):
        	images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        	labels[i] = labelArr[ind[i]]


    	filelabel = open(labelFile1, 'rb')
   	magic_nr, size = struct.unpack(">II", filelabel.read(8))
    	labelArr = pyarray("b", filelabel.read())
    	filelabel.close()

   	fimg = open(imageFile1, 'rb')
	magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
	img = pyarray("B", fimg.read())
    	fimg.close()

    	ind = [ k for k in range(size) if labelArr[k] in digits ]
   	N = len(ind)

    	images1 = zeros((N, rows, cols), dtype=uint8)
    	labels1 = zeros((N, 1), dtype=int8)
    	for i in range(len(ind)):
        	images1[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        	labels1[i] = labelArr[ind[i]]

	'''
	row,col,ch=images.shape
	gauss=np.random.randn(row,col,ch)
	gauss=gauss.reshape(row,col,ch)
	images=images+images*gauss
	row,col,ch=images1.shape
	gauss=np.random.randn(row,col,ch)
	gauss=gauss.reshape(row,col,ch)
	images1=images1+images1*gauss'''
    	return images/255, labels,images1,labels1
    	
    	
#creating i array of size 10 with all zeros except 1 at the correct output value index    	
def makeVec(i):
    arr = np.zeros((10, 1))
    arr[i] = 1.0
    return arr	


#creating a vector of 28 X 28 as input to our neural network
#and creating a vector of 10 for indicating label
def inputFormat(images,labels,images1,labels1):
	tr_inputs = [np.reshape(x, (784, 1)) for x in images]
	tr_results = [makeVec(y) for y in labels]
	ts_inputs = [np.reshape(x, (784, 1)) for x in images1]
	ts_results = [makeVec(y) for y in labels1]
	train_set=zip(tr_inputs,tr_results)
	test_set=zip(ts_inputs,ts_results)
	return train_set,test_set
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sig_delta(z):
    return sigmoid(z)*(1-sigmoid(z))
    

def updateWgt(batch,wgt,bs):
    	delB=[np.zeros(b.shape,dtype=float) for b in bs]
        delW=[np.zeros(w.shape,dtype=float) for w in wgt]
	nebB=[np.zeros(b.shape,dtype=float) for b in bs]
	nebW=[np.zeros(w.shape,dtype=float) for w in wgt]
    	for x,y in batch:
        	   list1=[]
                   list1.append(x)                                                     
                   list1.append((np.dot(wgt[0],x)+bs[0])/255)                                  
                   list1.append((np.dot(wgt[1],sigmoid(list1[1]))+bs[1])/255)                                 
		   B=[np.zeros(b.shape,dtype=float) for b in bs]
                   W=[np.zeros(w.shape,dtype=float) for w in wgt]
                   deltaK=(sigmoid(list1[2])-y)*sig_delta(list1[2])
                   B[1]=deltaK
       		   W[1]=np.dot(deltaK,sigmoid(list1[1]).transpose())
                   deltaJ=np.dot(wgt[1].transpose(),deltaK)*sig_delta(list1[1])
                   B[0]=deltaJ
                   W[0]=np.dot(deltaJ,list1[0].transpose())
        	   nebB=[b1+b2 for b1,b2 in zip(nebB,B)]
                   nebW=[w1+w2 for w1,w2 in zip(nebW,W)]
        delW=[w1-(2.0)*w2 for w1,w2 in zip(wgt,nebW)]
        delB=[b1-(2.0)*b2 for b1,b2 in zip(bs,nebB)]

    #print bn
        return delW,delB

def confusion(list1,list2):
	
	print ".................Confusion matrix................ "
	#print list1
	
	tp=[0,0,0,0,0,0,0,0,0,0]
	tn=[0,0,0,0,0,0,0,0,0,0]
	fp=[0,0,0,0,0,0,0,0,0,0]
	fn=[0,0,0,0,0,0,0,0,0,0]
	"""
	i=0
	while(i<10):
		#tp.append(0)
		#tn.append(0)
		fp.append(0)
		fn.append(0)
		i+=1
	"""
	confuse={}
	num=len(list1)
	i=0
    	while (i<num):
        	item1=list2[i]
        	if item1 not in confuse:
        		confuse[item1]={}
        	j=0	
        	while (j<num):
            		item2=list2[j]
            		if item2 not in confuse[item1]:
                		confuse[item1][item2]=0
			j+=1
		i+=1
	i=0
	while (i<num):
		item1=list1[i]
		item2=list2[i]
        	i+=1
        	confuse[item2][item1]+=1
		
        for i in confuse.keys():
        	tp[i]=confuse[i][i]
		print "class: ", i  ,"=",
       		for j in confuse[i].keys():
       			if(j!=i):
       				fn[i]=fn[i]+confuse[i][j]
       				fp[i]=fp[i]+confuse[j][i]
       				
             		print j,": ",confuse[i][j],"    ",
        	print ""
	i=0
	while( i<10):
		tn[i]=10000-tp[i]-fp[i]-fn[i]
		print "Accuracy for "+str(i)+"= ",
		temp=(float(tp[i]+tn[i])*100)/float(tp[i]+tn[i]+fp[i]+fn[i])
		print str(temp)+"% "
		print "precision for "+str(i)+"= ",
		if(tp[i]+fp[i]!=0):
			temp=(float(tp[i])*100)/float(tp[i]+fp[i])
		else:	
			temp=0.0
		print str(temp)+"% "
		print "Sensivity for "+str(i)+"= ",
		temp=(float(tp[i])*100)/float(tp[i]+fn[i])
		print str(temp)+"% "
		print "Specificity for "+str(i)+"= ",
		temp=(float(tn[i])*100)/float(tn[i]+fp[i])
		print str(temp)+"% "
		print
		i+=1





def feedforward(i,wgt,bs):
	i=sigmoid(np.dot(wgt[0],i)+bs[0])
	i=sigmoid(np.dot(wgt[1],i)+bs[1])
	return argmax(i)

def check(test_set,wgt,bs):
        hits=0
        list1=[]
        list2=[]
	for x,y in test_set:
		temp=feedforward(x,wgt,bs)
		list1.append(temp)
		list2.append(argmax(y))
		if temp==argmax(y):
			hits=hits+1
	print ""
	print ""
	print "accuracy: ", hits/100.0,"%"
	confusion(list1,list2)

		
images, labels,images1, labels1 = read_data()
train_set,test_set=inputFormat(images,labels,images1,labels1)
#print len(train_set)
#test_set=zip(images1,labels1)
#print len(test_set)
biases = [np.random.randn(y, 1) for y in [100,10]]
weights = [np.random.randn(x, y) for x, y in [[100,784],[10,100]]]
n=20

print "............updating weights............."
print ""

for i in range(3):
	random.shuffle(train_set)
	#batches =[train_set[j:j+10] for j in range(0,len(train_set),10)]
	#for batch in batches:
	for k in range(0,len(train_set),20):
		weights,biases=updateWgt(train_set[k:k+20],weights,biases)
	check(test_set,weights,biases)


def distance(a,b):
	temp=0
	for i in range(784):
		t=a.item(i)-b.item(i)
		temp=temp+(t*t)
	return math.sqrt(temp)


def neighbour(test_set,train_set):
	count=0
	for (a,b) in test_set:
		dist=0
		#print b
		clas=np.zeros((10, 1))
		for (x,y) in train_set:
			t= distance(a,x)
			#print t
			if(dist>t):
				clas=y
		if(argmax(b)==argmax(y)):
			count+=1
	print count
	eff=(float(count)/len(test_set))*100.0
	print "efficiency :", eff,"%"
			
test1=test_set
train1=train_set
#print len(train1), len(test1)
print ""
print ""
print "................1NN................"
print ""

#neighbour(test1,train1)


length=len(train_set)
offset=length/5
print ""
print ""
print "................5-Fold cross validation................"
print ""

'''valid_sets=[train_set[i:i+offset] for i in range(0,length,offset)]
#print len(valid_sets)
for i in range(5):
	valid_set=valid_sets[i]
	print len(valid_set)
	weights,biases=check(valid_set,weights,biases)
	#random.shuffle(train_set)
	#train_set1=train_set[0:(offset*4)]
	#print len(train_set1)'''
for k in xrange(0,length,offset):
	batch=train_set[k:k+offset]
	for i in range(5):
		random.shuffle(batch)
		m_batches=[batch[j:j+20] for j in range(0,offset,20)]
		for m_batch in m_batches:
			weights,biases=updateWgt(m_batch,weights,biases)
	check(test_set,weights,biases)
