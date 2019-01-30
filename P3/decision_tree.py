import numpy as np
from typing import List
from classifier import Classifier

class DecisionTree(Classifier):
	def __init__(self):
		self.clf_name = "DecisionTree"
		self.root_node = None

	def train(self, features: List[List[float]], labels: List[int]):
		# init.
		assert(len(features) > 0)
		self.feautre_dim = len(features[0])
		num_cls = np.max(labels)+1

		# build the tree
		self.root_node = TreeNode(features, labels, num_cls)
		if self.root_node.splittable:
			self.root_node.split()

		return
		
	def predict(self, features: List[List[float]]) -> List[int]:
		y_pred = []
		for feature in features:
			y_pred.append(self.root_node.predict(feature))
		return y_pred

	def print_tree(self, node=None, name='node 0', indent=''):
		if node is None:
			node = self.root_node
		print(name + '{')
		
		string = ''
		for idx_cls in range(node.num_cls):
			string += str(node.labels.count(idx_cls)) + ' '
		print(indent + ' num of sample / cls: ' + string)

		if node.splittable:
			print(indent + '  split by dim {:d}'.format(node.dim_split))
			for idx_child, child in enumerate(node.children):
				self.print_tree(node=child, name= '  '+name+'/'+str(idx_child), indent=indent+'  ')
		else:
			print(indent + '  cls', node.cls_max)
		print(indent+'}')


class TreeNode(object):
	def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
		self.features = features
		self.labels = labels
		self.children = []
		self.num_cls = num_cls

		count_max = 0
		for label in np.unique(labels):
			if self.labels.count(label) > count_max:
				count_max = labels.count(label)
				self.cls_max = label # majority of current node

		if len(np.unique(labels)) < 2:
			self.splittable = False
		else:
			self.splittable = True

		self.dim_split = None # the index of the feature to be split

		self.feature_uniq_split = None # the possible unique values of the feature to be split


	def split(self):
		def conditional_entropy(branches: List[List[int]]) -> float:
			'''
			branches: C x B array, 
					  C is the number of classes,
					  B is the number of branches
					  it stores the number of 
					  corresponding training samples 
					  e.g.
					              ○ ○ ○ ○
					              ● ● ● ●
					            ┏━━━━┻━━━━┓
				               ○ ○       ○ ○
				               ● ● ● ●
				               
				      branches = [[2,2], [4,0]]
			'''
			########################################################
			# TODO: compute the conditional entropy
			########################################################c
			Sum=np.sum(branches,axis=0)
			P=branches/Sum
			P1=P
			P1[P==0]=1
			Log=np.log2(P1)
			entropy=-np.sum(np.multiply(P,Log),axis=0)
			weight=Sum/(np.sum(branches))
			cond_entropy=np.sum(np.multiply(weight,entropy))
			return cond_entropy

		Min_entropy=float('inf')
		label=np.array(self.labels)
		get_lab=np.unique(label)
		label_class={}
		for ind,n in enumerate(get_lab):
			label_class[n]=ind
		#print ("features",self.features)
		#print ("labels",self.labels)
		for idx_dim in range(len(self.features[0])):
			Ind=[]
			############################################################
			# TODO: compare each split using conditional entropy
			#       find the best split
			############################################################
			Fea=np.array(self.features)[:,idx_dim]
			if None in Fea:
				continue
			diff_Fea=np.unique(Fea)#decide how many branches
			branches=np.zeros((self.num_cls,len(diff_Fea)))
			for i,elem in enumerate(diff_Fea):
				ind=np.where(Fea==elem)
				Ind.append(ind)
				Cla=label[ind]#the class that belong to the same branch
				un,count=np.unique(Cla,return_counts=True)# the branches location
				for b_ind,b_ele in enumerate(un):
					branches[label_class[b_ele]][i]=count[b_ind]
				branche=branches.tolist()
			#print (branche)
			entropy=conditional_entropy(branche)
			#print (entropy)
			if entropy<Min_entropy:
				Min_entropy=entropy
				self.dim_split=idx_dim
				self.feature_uniq_split=diff_Fea.tolist()
				Index=Ind# save the index of choosed sample
		############################################################
		# TODO: split the node, add child nodes
		############################################################
		feature=np.array(self.features,dtype=object)
		feature[:,self.dim_split]=None# when the feature is used to split, it never consider
		#print ('feature',feature)
		#print ('dim',self.dim_split)
		for num_child in range(len(self.feature_uniq_split)): # construct every child branches/node
			new_feature=feature[Index[num_child]].tolist()# the choosed feature of this child node
			new_label=label[Index[num_child]].tolist()# the choosed feature's label of this child node
			#print ("new",new_feature,new_label)
			child_node=TreeNode(new_feature,new_label,self.num_cls)
			if np.count_nonzero(new_feature)==0:#no more features or no example
				child_node.splittable=False
			self.children.append(child_node)

		# split the child nodes
		for child in self.children:
			if child.splittable:
				child.split()

		return

	def predict(self, feature: List[int]) -> int:
		if self.splittable:
			# print(feature)
			idx_child = self.feature_uniq_split.index(feature[self.dim_split])
			return self.children[idx_child].predict(feature)
		else:
			return self.cls_max



