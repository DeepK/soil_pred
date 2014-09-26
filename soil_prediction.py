import pandas as pd, numpy as np, sys, argparse, random
from sklearn.cross_validation import KFold
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.manifold import LocallyLinearEmbedding as LLE
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.svm import SVR

def get_data():

	#Read file
	train=pd.read_csv('training.csv')
	labels=labels = train[['Ca','P','pH','SOC','Sand']].values
	#Don't need the labels in the train set
	train.drop(['Ca','P','pH','SOC','Sand','PIDN'],axis=1,inplace=True)
	#Must replace strings with ints before converting to numpy array
	train.replace('Topsoil',1,inplace=True)
	train.replace('Subsoil',0,inplace=True)

	#Similar process
	test=pd.read_csv('sorted_test.csv')
	test.drop(['PIDN'],axis=1,inplace=True)
	test.replace('Topsoil',1,inplace=True)
	test.replace('Subsoil',0,inplace=True)

	return train,labels,test

def err(real,pred):

	#Returns the mean columnwise root mean square error
	return np.mean(np.mean((real-pred)**2,axis=0)**0.5)

def tp_regressor(train,labels,test,regressor,regularization,max_iter=50000):
	"""
		Function to train regressor on each feature vector by itself
		and predict values on 'test'

		parameters:
			regularization : C (for SVR) or alpha (for regression)
			max_iter : for regression
	"""

	pr=[]
	for i in xrange(5):

		#### TODO:Add more regressors here #####

		if regressor=='Lasso':
			r=Lasso(alpha=regularization,max_iter=max_iter)
		elif regressor=='SVR':
			r=SVR(C=regularization)
		elif regressor=='Ridge':
			r=Ridge(alpha=regularization)

		#Fit
		r.fit(train,labels[:,i])
		#Predict
		pr.append(r.predict(test).tolist())

	return pr

def predict_values(train,labels,test,regressor,regularization,random_subspace,num_regressor):

	#This is where the regressor is trained and values predicted for
	#CV or submission

	pred=np.zeros((5,test.shape[0]))
	print "Training and predicting",

	if random_subspace>0:
		print "with random subspace method;",random_subspace,"fraction of features",num_regressor,"number of regressors"
		for _ in xrange(num_regressor):
			smpl=random.sample(range(train.shape[1]),int(random_subspace*train.shape[1]))
			pr=tp_regressor(train[:,smpl],labels,test[:,smpl],regressor,regularization)
			pred=pred+np.asarray(pr,dtype=float)

		pred=pred/num_regressor
	else:
		print
		pr=tp_regressor(train,labels,test,regressor,regularization)
		pred=pred+np.asarray(pr,dtype=float)

	return pred.T

def btstrap(train,labels,test,regressor,regularization,random_subspace,num_regressor,bootstrap):

	print "Booststrapping:"
	pred=np.zeros((test.shape[0],5))
	for _ in xrange(bootstrap):
		smpl=np.random.random_integers(0,train.shape[0]-1,train.shape[0]-1)
		pred=pred+predict_values(train[smpl,:],labels[smpl,:],test,regressor,regularization,random_subspace,num_regressor)
	pred=pred/bootstrap
	return pred

def cross_validate(train_n,labels_n,random_subspace,num_regressor,bootstrap,folds=5,dim_reduce='',regressor='SVR',regularization=3000):

	if dim_reduce=='lle':
		print "LLE.."
		le=LLE(n_neighbors=100,n_components=30)
		train_n=le.fit_transform(train_n)

	elif dim_reduce=='pca':
		print "PCA.."
		p=PCA(n_components=17)
		train_n=p.fit_transform(train_n)

		## To select optimum number of components ##
		#print np.where(np.cumsum(p.explained_variance_ratio_)>0.99)[0][0]+1

	elif dim_reduce=='kpca':
		print "Kernel PCA.."
		p=KernelPCA(kernel='rbf')
		train_n=p.fit_transform(train_n)

	print folds,"Fold Cross validation..\n"
	kf=KFold(len(train_n),n_folds=folds)

	er=[]
	f=1
	for tr_in,tst_in in kf:
		print "Fold :",f

		train_X,trainlabel=train_n[tr_in],labels_n[tr_in]
		test_X,testlabel=train_n[tst_in],labels_n[tst_in]

		if bootstrap>0:
			pred=btstrap(train_X,trainlabel,test_X,regressor,regularization,random_subspace,num_regressor,bootstrap)
		else:
			pred=predict_values(train_X,trainlabel,test_X,regressor,regularization,random_subspace,num_regressor)

		er.append(err(testlabel,pred))
		f+=1

	return er,np.mean(er),np.std(er)

def finalize(train_n,labels_n,test_n,random_subspace,num_regressor,bootstrap,regressor='SVR',regularization=3000):

	#Writes output for submission

	if bootstrap>0:
		pred=btstrap(train_n,labels_n,test_n,regressor,regularization,random_subspace,num_regressor,bootstrap)
	else:
		pred=predict_values(train_n,labels_n,test_n,regressor,regularization,random_subspace,num_regressor)

	print "Writing to file..\n"
	sample=pd.read_csv('sample_submission.csv')
	sample['Ca']=pred[:,0]
	sample['P']=pred[:,1]
	sample['pH']=pred[:,2]
	sample['SOC']=pred[:,3]
	sample['Sand']=pred[:,4]

	sample.to_csv('submission.csv',index=False)

if __name__ == '__main__':

	#Adding arguments to be accepted and parsed
	parser=argparse.ArgumentParser()
	parser.add_argument("-r","--regularization",type=float,default=3000.0,help="regularization parameter (C for SVR, lambda for regression)")
	parser.add_argument("-g","--regressor",default="SVR",help="regressor name (SVR or Ridge or Lasso)")
	parser.add_argument("-o","--option",default='v',help="'v' for k fold CV, 's' for producing submission file")
	parser.add_argument("-d","--dimred",default='',help="'pca' 'lle' or 'kpca' for dimensionality reduction")
	parser.add_argument("-f","--folds",type=int,default=10,help="number of folds for CV")
	parser.add_argument("-b","--randomsub",type=float,default=0.5,help="fraction of features to use")
	parser.add_argument("-nr","--numregressor",type=int,default=10,help="number of regressors to train")
	parser.add_argument("-bt","--numboot",type=int,default=10,help="number of bootstraps")
	args=parser.parse_args()

	train,labels,test=get_data()

	#Convert to numpy arrays
	train_n=np.asarray(train,dtype=float)
	test_n=np.asarray(test,dtype=float)
	labels_n=np.asarray(labels,dtype=float)

	print "regressor :",args.regressor
	print "Regularization :",args.regularization
	print

	if args.option=='v':
		e,m,s=cross_validate(train_n,labels_n,folds=args.folds,dim_reduce=args.dimred,regressor=args.regressor,regularization=args.regularization,random_subspace=args.randomsub,num_regressor=args.numregressor,bootstrap=args.numboot)
		print "Errors :",e
		print "Mean :",m
		print "Std :",s

	elif args.option=='s':
		finalize(train_n,labels_n,test_n,args.randomsub,args.numregressor,args.numboot,args.regressor,args.regularization)

	else:
		raise Exception("Wrong option, see help")
