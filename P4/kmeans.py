import numpy as np


class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int) 
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x):
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        '''
        Finds n_cluster in the data x
        params:
        x - N X D numpy array
        returns:
        A tuple
        (centroids a n_cluster X D numpy array, y a size (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates an Int)
        Note: Number of iterations is the number of time you update the assignment
        '''
        np.random.seed(42)
        N, D = x.shape

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
        #raise Exception(
            #'Implement fit function in KMeans class (filename: kmeans.py)')
        #initialization 
        center=x[np.random.choice(N,self.n_cluster)]
        J=np.inf#distortion objective
        #repeat
        i=0
        while i<=self.max_iter:
            distance=np.zeros((N,self.n_cluster))
            for k in range(self.n_cluster):
                distance[:,k]=np.linalg.norm(x-center[k],axis=1)
            #print (distance)
            R=np.argmin(distance,axis=1)
            #print (R)
            J_new=np.mean(np.power(np.linalg.norm(x-center[R],axis=1),2))
            i=i+1
            if abs(J-J_new)<=self.e:
                break
            else:
                J=J_new
                for k in range(self.n_cluster):
                    ind=np.where(R==k)
                    if ind[0].size==0:
                        continue
                    else:
                        center[k]=np.mean(x[ind],axis=0)
        return center, R, i        
        
        # DONOT CHANGE CODE BELOW THIS LINE

class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int) 
            e - error tolerance (Float) 
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x, y):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by 
                    majority voting ((N,) numpy array) 
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        #raise Exception(
            #'Implement fit function in KMeansClassifier class (filename: kmeans.py)')
        k_means=KMeans(n_cluster=self.n_cluster,max_iter=self.max_iter,e=self.e)
        centroids,cluster,i=k_means.fit(x)
        centroid_labels=np.zeros(self.n_cluster)
        for k in range(self.n_cluster):
            ind=np.where(cluster==k)
            if ind[0].size==0:
                centroid_labels[k]=0
            else:
                centroid_labels[k]=np.argmax(np.bincount(y[ind]))


        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(
            self.n_cluster)

        assert self.centroids.shape == (self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(
            self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function

            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        #raise Exception(
            #'Implement predict function in KMeansClassifier class (filename: kmeans.py)')
        Dist=np.zeros((N,self.n_cluster))
        labels=np.zeros(N,)
        for k in range(self.n_cluster):
            Dist[:,k]=np.linalg.norm(x-self.centroids[k],axis=1)
        R_pre=np.argmin(Dist,axis=1)
        labels=self.centroid_labels[R_pre]

        # DONOT CHANGE CODE BELOW THIS LINE
        return labels

