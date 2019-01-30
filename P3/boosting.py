import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
  # Boosting from pre-defined classifiers
	def __init__(self, clfs: Set[Classifier], T=0):
		self.clfs = clfs      # set of weak classifiers to be considered
		self.num_clf = len(clfs)
		if T < 1:
			self.T = self.num_clf
		else:
			self.T = T

		self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
		self.betas = []       # list of weights beta_t for t=0,...,T-1
		return

	@abstractmethod
	def train(self, features: List[List[float]], labels: List[int]):
		return

	def predict(self, features: List[List[float]]) -> List[int]:
		'''
		Inputs:
		- features: the features of all test examples
   
		Returns:
		- the prediction (-1 or +1) for each example (in a list)
		'''
		########################################################
		# TODO: implement "predict"
		########################################################
		N=len(features)
		l=len(self.betas)
		H=np.zeros(N,)
		for bt,ht in zip(self.betas,self.clfs_picked):
			w_ht=bt*np.array(ht.predict(features))
			H=H+w_ht
		for n in range(N):
			if H[n]>=0:
				H[n]=1
			else:
				H[n]=-1
		return H.tolist()


class AdaBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "AdaBoost"
		return
		
	def train(self, features: List[List[float]], labels: List[int]):
		'''
		Inputs:
		- features: the features of all examples
		- labels: the label of all examples
   
		Require:
		- store what you learn in self.clfs_picked and self.betas
		'''
		############################################################
		# TODO: implement "train"
		############################################################
		#step 1
		N=len(features)
		labels=np.array(labels)
		D=np.array([1/N]*N)
		#step 2
		for t in range(self.T):
			e_t=float('inf')
			for a in self.clfs:
				arr=np.array(a.predict(features))
				diff_ind=np.where(labels!=arr)
				Fin=np.sum(D[diff_ind])
				if Fin<e_t:
					e_t=Fin
					h_t=arr#find min weighted loss,step 3
					h_t_t=a#step 4, find h_t
			self.clfs_picked.append(h_t_t)
			beta_t=0.5*np.log((1-e_t)/e_t)#step 5
			self.betas.append(beta_t)
			d_ind=np.where(h_t!=labels)
			s_ind=np.where(h_t==labels)
			D[s_ind]=np.multiply(D[s_ind],np.exp(-beta_t))
			D[d_ind]=np.multiply(D[d_ind],np.exp(beta_t))#step 6
			Sum=np.sum(D)
			D=D/Sum #step 7				
		
	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)



	