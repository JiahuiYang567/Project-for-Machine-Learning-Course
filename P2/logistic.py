from __future__ import division, print_function

import numpy as np
import math

#######################################################################
# Replace TODO with your code
#######################################################################
def sign(X,w,N):
	y_pre=np.dot(X,w)
	for i in range(N):
		if (y_pre[i]>=0):
			y_pre[i]=1
		else:
			y_pre[i]=0
	return y_pre
def binary_train(X, y, w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:       
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data
    - step_size: step size (learning rate)
	- max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of logistic regression
    - b: scalar, which is the bias of logistic regression
    Find the optimal parameters w and b for inputs X and y.
    Use the *average* of the gradients for all training examples
	multiplied by the step_size to update parameters.
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2

    w=np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0


    """
    TODO: add your code here
    """
    w=np.insert(w,0,b,axis=0)
    One=np.ones(N,)
    X=np.insert(X,0,One,axis=1)
    D=D+1
    def Ave(X,N,D,y,w,b):
    	Grad=np.zeros((D,N))
    	z=sigmoid(np.dot(X,w))-y
    	Grad=np.transpose(np.multiply(z.reshape(N,1),X))
    	grad=np.mean(Grad,axis=1)
    	return grad

    i=1
    while (i<=max_iterations) and (sign(X,w,N)!=y).any():
    	grad=Ave(X,N,D,y,w,b)
    	w=w-step_size*grad
    	i=i+1
    b=w[0]
    w=np.delete(w,0,axis=0)
    D=D-1
    assert w.shape == (D,)
    return w, b


def binary_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    
    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape
    preds = np.zeros(N) 
    """
    TODO: add your code here
    """ 
    One=np.ones(N,)
    X=np.insert(X,0,One,axis=1)
    w=np.insert(w,0,b,axis=0)
    preds=sign(X,w,N)
    assert preds.shape == (N,) 
    return preds


def multinomial_train(X, y, C, 
                     w0=None, 
                     b0=None, 
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data
    - C: number of classes in the data
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where 
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes

    Implement multinomial logistic regression for multiclass 
    classification. Again use the *average* of the gradients for all training 
	examples multiplied by the step_size to update parameters.
	
	You may find it useful to use a special (one-hot) representation of the labels, 
	where each label y_i is represented as a row of zeros with a single 1 in
    the column, that corresponds to the class y_i.
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0


    """
    TODO: add your code here
    """
    w=np.insert(w,0,b,axis=1)
    One=np.ones(N,)
    X=np.insert(X,0,One,axis=1)
    D=D+1
    def AVE_mul(X,N,D,w,C):
        Grad=np.zeros((N,C,D))
        soft_res=np.transpose(softmax(X,w,N))#C*N
        z=np.zeros((C,N))
        for n in range(N):
            z[y[n]][n]=1
        v=soft_res-z #probability of update matrix(C*N)
        for n in range(N):
            Grad[n]=np.dot(v[:,n].reshape(C,1),X[n,:].reshape(1,D))
        ave_grad=np.mean(Grad,axis=0)# C*D
        return ave_grad #C*D
    y_pred=softmax(X,w,N)
    y_Mul=np.argmax(y_pred,axis=1)
    i=1
    while i<=max_iterations and (y_Mul!=y).any():
        Grad=AVE_mul(X,N,D,w,C)
        w=w-step_size*Grad
        i=i+1
        y_pred=softmax(X,w,N)
        y_Mul=np.argmax(y_pred,axis=1)
            
    b=w[:,0]
    w=np.delete(w,0,axis=1)
    D=D-1   

    assert w.shape == (C, D)
    assert b.shape == (C,)
    return w, b

def softmax(X,w,N):
	m=np.dot(X,np.transpose(w))# N*C
	Max=np.amax(m,axis=1)# N*1
	m_1=m-Max.reshape(N,1) #N*C
	Exp=np.exp(m_1)
	de=np.sum(Exp,axis=1)
	Prob=Exp/(de.reshape(N,1)) #N*C
	return Prob

def multinomial_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier
    - b: bias terms of the trained multinomial classifier
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes

    Make predictions for multinomial classifier.
    """
    N, D = X.shape
    C = w.shape[0]
    preds = np.zeros(N) 

    """
    TODO: add your code here
    """ 
    One=np.ones(N,) 
    X=np.insert(X,0,One,axis=1)
    w=np.insert(w,0,b,axis=1)
    y_pred=softmax(X,w,N)
    preds=np.argmax(y_pred,axis=1)

    assert preds.shape == (N,)
    return preds


def OVR_train(X, y, C, w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array, 
    indicating the labels of each training point
    - C: number of classes in the data
    - w0: initial value of weight matrix
    - b0: initial value of bias term
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: a C-by-D weight matrix of OVR logistic regression
    - b: bias vector of length C

    Implement multiclass classification using one-versus-rest with binary logistic 
	regression as the black-box. Recall that the one-versus-rest classifier is 
    trained by training C different classifiers. 
    """
    N, D = X.shape
    
    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0

    """
    TODO: add your code here
    """
    def AVE_OVR(X,w,y,C,D,N):
    	Grad_OVR=np.zeros((C,D))
    	Y=np.zeros((N,C))
    	for i in range(N):
    		Y[i][y[i]]=1
    	z=sigmoid(np.dot(X,np.transpose(w)))-Y
    	for i in range(C):
    		Grad_OVR[i,:]=np.mean(np.multiply(z[:,i].reshape(N,1),X),axis=0)
    	return Grad_OVR

    One=np.ones(N,)
    X=np.insert(X,0,One,axis=1)
    w=np.insert(w,0,b,axis=1)
    D=D+1
    i=1
    Pro_OVR=softmax(X,w,N)
    Pred_OVR=np.argmax(Pro_OVR,axis=1) #find max.index in every row
    while (i<=max_iterations) and (Pred_OVR!=y).any():
    	grad_OVR=AVE_OVR(X,w,y,C,D,N)
    	w=w-step_size*grad_OVR
    	Pred_OVR=softmax(X,w,N)
    	Pred_OVR=np.argmax(Pro_OVR,axis=1)
    	i=i+1
    b=w[:,0]
    w=np.delete(w,0,axis=1)
    D=D-1
    assert w.shape == (C, D), 'wrong shape of weights matrix'
    assert b.shape == (C,), 'wrong shape of bias terms vector'
    return w, b


def OVR_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained OVR model
    - b: bias terms of the trained OVR model
    
    Returns:
    - preds: vector of class label predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes.

    Make predictions using OVR strategy and probability predictions from binary
    classifiers. 
    """
    N, D = X.shape
    C = w.shape[0]
    preds = np.zeros(N) 
    
    """
    TODO: add your code here
    """
    One=np.ones(N,)
    X=np.insert(X,0,One,axis=1)
    w=np.insert(w,0,b,axis=1)
    Pred_prob=softmax(X,w,N)
    preds=np.argmax(Pred_prob,axis=1)

    assert preds.shape == (N,)
    return preds


#######################################################################
# DO NOT MODIFY THE CODE BELOW 
#######################################################################

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def accuracy_score(true, preds):
    return np.sum(true == preds).astype(float) / len(true)

def run_binary():
    from data_loader import toy_data_binary, \
                            data_loader_mnist 

    print('Performing binary classification on synthetic data')
    X_train, X_test, y_train, y_test = toy_data_binary()
        
    w, b = binary_train(X_train, y_train)
    
    train_preds = binary_predict(X_train, w, b)
    preds = binary_predict(X_test, w, b)
    print('train acc: %f, test acc: %f' % 
            (accuracy_score(y_train, train_preds),
             accuracy_score(y_test, preds)))
    
    print('Performing binary classification on binarized MNIST')
    X_train, X_test, y_train, y_test = data_loader_mnist()

    binarized_y_train = [0 if yi < 5 else 1 for yi in y_train] 
    binarized_y_test = [0 if yi < 5 else 1 for yi in y_test] 
    
    w, b = binary_train(X_train, binarized_y_train)
    train_preds = binary_predict(X_train, w, b)
    preds = binary_predict(X_test, w, b)
    print('train acc: %f, test acc: %f' % 
            (accuracy_score(binarized_y_train, train_preds),
             accuracy_score(binarized_y_test, preds)))

def run_multiclass():
    from data_loader import toy_data_multiclass_3_classes_non_separable, \
                            toy_data_multiclass_5_classes, \
                            data_loader_mnist 
    
    datasets = [(toy_data_multiclass_3_classes_non_separable(), 
                        'Synthetic data', 3), 
                (toy_data_multiclass_5_classes(), 'Synthetic data', 5), 
                (data_loader_mnist(), 'MNIST', 10)]

    for data, name, num_classes in datasets:
        print('%s: %d class classification' % (name, num_classes))
        X_train, X_test, y_train, y_test = data
        
        print('One-versus-rest:')
        w, b = OVR_train(X_train, y_train, C=num_classes)
        train_preds = OVR_predict(X_train, w=w, b=b)
        preds = OVR_predict(X_test, w=w, b=b)
        print('train acc: %f, test acc: %f' % 
            (accuracy_score(y_train, train_preds),
             accuracy_score(y_test, preds)))
    
        print('Multinomial:')
        w, b = multinomial_train(X_train, y_train, C=num_classes)
        train_preds = multinomial_predict(X_train, w=w, b=b)
        preds = multinomial_predict(X_test, w=w, b=b)
        print('train acc: %f, test acc: %f' % 
            (accuracy_score(y_train, train_preds),
             accuracy_score(y_test, preds)))


if __name__ == '__main__':
    
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--type", )
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.output:
            sys.stdout = open(args.output, 'w')

    if not args.type or args.type == 'binary':
        run_binary()

    if not args.type or args.type == 'multiclass':
        run_multiclass()
        