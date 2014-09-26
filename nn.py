import soil_prediction
import numpy as np
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import LinearLayer
from pybrain.supervised.trainers import BackpropTrainer

if __name__ == '__main__':

	train,labels,test=soil_prediction.get_data()

	#Convert to numpy arrays
	train_n=np.asarray(train,dtype=float)
	test_n=np.asarray(test,dtype=float)
	labels_n=np.asarray(labels,dtype=float)

	for i in xrange(train_n.shape[1]):
		train_n[:,i]=train_n[:,i]-np.mean(train_n[:,i])
		train_n[:,i]=train_n[:,i]/np.std(train_n[:,i])

	DS=SupervisedDataSet(train_n.shape[1],5)
	for i in xrange(train_n.shape[0]):
		DS.addSample(train_n[i],labels_n[i])

	net=buildNetwork(train_n.shape[1],500,5,hiddenclass=LinearLayer)

	trainer=BackpropTrainer(net,DS)
	for _ in xrange(10):
		er=trainer.train()
		print er
