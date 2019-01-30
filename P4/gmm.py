import numpy as np
from kmeans import KMeans

class GMM():
    '''
        Fits a Gausian Mixture model to the data.

        attrs:
            n_cluster : Number of mixtures (Int)
            e : error tolerance (Float) 
            max_iter : maximum number of updates (Int)
            init : initialization of means and variance
                Can be 'random' or 'kmeans' 
            means : means of Gaussian mixtures (n_cluster X D numpy array)
            variances : variance of Gaussian mixtures (n_cluster X D X D numpy array) 
            pi_k : mixture probabilities of different component ((n_cluster,) size numpy array)
    '''

    def __init__(self, n_cluster, init='k_means', max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.e = e
        self.max_iter = max_iter
        self.init = init
        self.means = None
        self.variances = None
        self.pi_k = None

    def fit(self, x):
        '''
            Fits a GMM to x.

            x: is a NXD size numpy array
            updates:
                self.means
                self.variances
                self.pi_k
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'

        np.random.seed(42)
        N, D = x.shape

        if (self.init == 'k_means'):
            # TODO
            # - comment/remove the exception
            # - initialize means using k-means clustering
            # - compute variance and pi_k (see P4.pdf)

            # DONOT MODIFY CODE ABOVE THIS LINE
            #raise Exception(
                #'Implement initialization of variances, means, pi_k using k-means')
            kmeans=KMeans(n_cluster=self.n_cluster,max_iter=self.max_iter,e=self.e)
            self.means,cluster,i=kmeans.fit(x)#initialization
            Y=np.zeros((N,self.n_cluster))
            Y[range(N),cluster]=1
            #print ("1",Y)
            #Y=np.eye(self.n_cluster)[cluster]
            #print ("2",Y)
            self.variances=np.zeros((self.n_cluster,D,D))
            self.pi_k=np.zeros(self.n_cluster)
            for k in range(self.n_cluster):
                var_1=np.multiply(np.eye(D),np.dot(np.transpose(Y[:,k].reshape(N,1)*(x-self.means[k])),(x-self.means[k])))#binary yik
                self.variances[k,:,:]=var_1/(np.sum(Y[:,k]))
                self.pi_k[k]=np.sum(Y[:,k])/N
                
            # DONOT MODIFY CODE BELOW THIS LINE

        elif (self.init == 'random'):
            # TODO
            # - comment/remove the exception
            # - initialize means randomly
            # - initialize variance to be identity and pi_k to be uniform

            # DONOT MODIFY CODE ABOVE THIS LINE
            #raise Exception(
                #'Implement initialization of variances, means, pi_k randomly')
            self.means=np.random.rand(self.n_cluster,D)
            var=np.eye(D)
            self.variances=np.zeros((self.n_cluster,D,D))
            self.variances[range(self.n_cluster),:,:]=var
            self.pi_k=np.array([1/self.n_cluster]*self.n_cluster)
            Y=np.zeros((N,self.n_cluster))
            # DONOT MODIFY CODE BELOW THIS LINE

        else:
            raise Exception('Invalid initialization provided')

        # TODO
        # - comment/remove the exception
        # - Use EM to learn the means, variances, and pi_k and assign them to self
        # - Update until convergence or until you have made self.max_iter updates.
        # - Return the number of E/M-Steps executed (Int) 
        # Hint: Try to separate E & M step for clarity
        # DONOT MODIFY CODE ABOVE THIS LINE
        #raise Exception('Implement fit function (filename: gmm.py)')
        loglike=self.compute_log_likelihood(x)
        i=0
        while i<=self.max_iter:
            #E step
            num=np.zeros((N,self.n_cluster))
            for k in range(self.n_cluster):
                while np.linalg.matrix_rank(self.variances[k,:,:])<D:
                    self.variances[k,:,:]=self.variances[k,:,:]+np.multiply(np.power(0.1,3),np.eye(D))
                inv=np.linalg.inv(self.variances[k,:,:])
                c=np.multiply(np.power(2*np.pi,D),np.linalg.det(self.variances[k,:,:]))
                a_1=np.dot((x-self.means[k,:]),inv)
                a_2=np.sum(np.multiply(a_1,(x-self.means[k,:])),axis=1)#(N*N)
                p=np.exp(-0.5*a_2)/np.sqrt(c)#(N)
                num[:,k]=self.pi_k[k]*p
            de=np.sum(num,axis=1)#N
            Y=num/de.reshape(N,1)
            #M step
            for k in range(self.n_cluster):
                nu_mean=np.sum(np.multiply(Y[:,k].reshape(N,1),x),axis=0)
                de_mean=np.sum(Y[:,k])
                self.means[k,:]=nu_mean/de_mean#update means
                nu_var=np.dot(np.multiply(np.transpose(x-self.means[k]),Y[:,k]),(x-self.means[k]))
                self.variances[k,:,:]=nu_var/(np.sum(Y[:,k]))
                self.pi_k[k]=de_mean/N
            loglike_new=self.compute_log_likelihood(x)
            i=i+1
            if np.abs(loglike-loglike_new)<=self.e:
                break
            else:
                loglike=loglike_new
        return i

        # DONOT MODIFY CODE BELOW THIS LINE

		
    def sample(self, N):
        '''
        sample from the GMM model

        N is a positive integer
        return : NXD array of samples

        '''
        assert type(N) == int and N > 0, 'N should be a positive integer'
        np.random.seed(42)
        if (self.means is None):
            raise Exception('Train GMM before sampling')

        # TODO
        # - comment/remove the exception
        # - generate samples from the GMM
        # - return the samples

        # DONOT MODIFY CODE ABOVE THIS LINE
        #raise Exception('Implement sample function in gmm.py')
        D=len(self.means[0])
        samples=np.zeros((N,D))
        #choose a gaussian according to pi_k
        K=np.random.choice(self.n_cluster,N,p=self.pi_k)#total N k,it means assign N sample to k clusters
        for ind, k in enumerate(K):
            samples[ind,:]=np.random.multivariate_normal(self.means[k,:],self.variances[k,:,:])
        # DONOT MODIFY CODE BELOW THIS LINE
        return samples        

    def compute_log_likelihood(self, x, means=None, variances=None, pi_k=None):
        '''
            Return log-likelihood for the data

            x is a NXD matrix
            return : a float number which is the log-likelihood of data
        '''
        assert len(x.shape) == 2,  'x can only be 2 dimensional'
        if means is None:
            means = self.means
        if variances is None:
            variances = self.variances
        if pi_k is None:
            pi_k = self.pi_k    
        # TODO
        # - comment/remove the exception
        # - calculate log-likelihood using means, variances and pi_k attr in self
        # - return the log-likelihood (Float)
        # Note: you can call this function in fit function (if required)
        # DONOT MODIFY CODE ABOVE THIS LINE
        #raise Exception('Implement compute_log_likelihood function in gmm.py')
        N,D=x.shape
        num=np.zeros((N,self.n_cluster))
        for k in range(self.n_cluster):
            while np.linalg.matrix_rank(variances[k,:,:])<D:
                variances[k,:,:]=variances[k,:,:]+np.multiply(np.power(0.1,3),np.eye(D))
            inv=np.linalg.inv(variances[k,:,:])
            c=np.multiply(np.power(2*np.pi,D),np.linalg.det(variances[k,:,:]))
            a_1=np.dot((x-means[k,:]),inv)
            a_2=np.sum(np.multiply(a_1,(x-means[k,:])),axis=1)#(N*N)
            p=np.exp(-0.5*a_2)/np.sqrt(c)#(N)
            num[:,k]=self.pi_k[k]*p
        de=np.sum(num,axis=1)#N
        Sum_n=np.sum(np.log(de))
        log_likelihood=float(Sum_n)
        # DONOT MODIFY CODE BELOW THIS LINE
        return log_likelihood
    class Gaussian_pdf():
        def __init__(self,mean,variance):
            self.mean = mean
            self.variance = variance
            self.c = None
            self.inv = None
            '''
                Input: 
                    Means: A 1 X D numpy array of the Gaussian mean
                    Variance: A D X D numpy array of the Gaussian covariance matrix
                Output: 
                    None: 
            '''
            # TODO
            # - comment/remove the exception
            # - Set self.inv equal to the inverse the variance matrix (after ensuring it is full rank - see P4.pdf)
            # - Set self.c equal to sqrt(((2pi)^D) * det(variance)) (after ensuring the variance matrix is full rank)
            # Note you can call this class in compute_log_likelihood and fit
            # DONOT MODIFY CODE ABOVE THIS LINE
            #raise Exception('Impliment Guassian_pdf __init__')
            D=len(self.variance)
            while np.linalg.matrix_rank(self.variance)!=D:
                self.variance=self.variance+np.multiply(np.power(0.1,3),np.eye(D))
            self.inv=np.linalg.inv(self.variance)
            self.c=np.multiply(np.power(2*np.pi,D),np.linalg.det(self.variance))
            # DONOT MODIFY CODE BELOW THIS LINE

        def getLikelihood(self,x):
            '''
                Input: 
                    x: a 1 X D numpy array representing a sample
                Output: 
                    p: a numpy float, the likelihood sample x was generated by this Gaussian
                Hint: 
                    p = e^(-0.5(x-mean)*(inv(variance))*(x-mean)')/sqrt(c)
                    where ' is transpose and * is matrix multiplication
            '''
            #TODO
            # - Comment/remove the exception
            # - Calculate the likelihood of sample x generated by this Gaussian
            # Note: use the described implementation of a Gaussian to ensure compatibility with the solutions
            # DONOT MODIFY CODE ABOVE THIS LINE
            #raise Exception('Impliment Guassian_pdf getLikelihood')
            D=len(x)
            a_1=np.dot((x-self.mean).reshape(1,D),self.inv)
            a_2=-0.5*np.dot(a_1,(x-self.mean).reshape(D,1))
            p=np.exp(a_2)/np.sqrt(self.c)
            # DONOT MODIFY CODE BELOW THIS LINE
            return p

    