
import numpy as np
import scipy.linalg as la
from mne.preprocessing.xdawn import _fit_xdawn
from mne.decoding import CSP as BaseCSP


class TRCA():
    def __init__(self,n_components=2):
        self.n_components = n_components

    def fit(self,X,y):
        """
        Parameters
        ----------
        X : ndarray, shape (n_epochs, n_channels, n_times)
            The data on which to estimate the CSP.
        y : array, shape (n_epochs,)
            The class for each epoch.
        Returns
        -------
        self : instance of TRCA
            Returns the modified instance.
        """
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        filter = []
        evokeds = []
        for classINX in range(n_classes):

            this_class_data = X[y==classINX]
            evoked = np.mean(this_class_data,axis=0)
            evokeds.append(evoked)
            weight = self.computer_trca_weight(this_class_data)
            filter.append(weight)

        self.filter = filter
        self.evokeds = evokeds
        return self

    def transform(self,X):
        """
        Parameters
        ----------
        X : array, shape (n_epochs, n_channels, n_times)
            The data.

        Returns
        -------
        X : ndarray
            shape is (n_epochs, n_sources, n_times).
        """
      
        
        enhanced = []
        for classINX in range(len(self._classes)):
            X_filtered = np.dot(self.filter[classINX][:,:self.n_components].T, X)
            X_filtered = X_filtered.transpose((1, 0, 2))
            X_filtered = np.stack(X_filtered[i].ravel() for i in range(X.shape[0]))
            enhanced.append(X_filtered)
        enhanced = np.concatenate(enhanced,axis=-1)

        return enhanced

    def computer_trca_weight(self,eeg):
        """
        Input:
            eeg : Input eeg data (# of targets, # of channels, Data length [sample])
        Output:
            W : Weight coefficients for electrodes which can be used as a spatial filter.           
        """
        epochNUM,self.channelNUM,_ = eeg.shape

        S = np.zeros((self.channelNUM,self.channelNUM))
        
        for epoch_i in range(epochNUM):
            x1 = np.squeeze(eeg[epoch_i,:,:])
            x1 = x1 - np.mean(x1,axis=1,keepdims=True)
            for epoch_j in range(epoch_i+1,epochNUM):
                x2 = np.squeeze(eeg[epoch_j,:,:])
                x2 = x2 - np.mean(x2,axis=1,keepdims=True)
                S = S + np.dot(x1,x2.T)+ np.dot(x2,x1.T)
        UX = np.stack(eeg[:,i,:].ravel() for i in range(self.channelNUM))
        UX = UX - np.mean(UX,axis=1,keepdims=True)
        Q = np.dot(UX,UX.T)

        _,W = la.eig(S,Q)
        return W


class Xdawn():
    def __init__(self,n_components=2):
        self.n_components = n_components

    def fit(self,X,y):
        """
        Parameters
        ----------
        X : ndarray, shape (n_epochs, n_channels, n_times)
            The data on which to estimate the CSP.
        y : array, shape (n_epochs,)
            The class for each epoch.
        Returns
        -------
        self : instance of TRCA
            Returns the modified instance.
        """
        self.channelNUM = X.shape[1]
        self._classes = np.unique(y)
        
        filters,_,evokeds = _fit_xdawn(X,y,n_components=self.channelNUM)
        filters = filters.T
        self.filter = np.split(filters,indices_or_sections=2,axis=1)
        self.evokeds = evokeds
        return self

    def transform(self,X):
        """
        Parameters
        ----------
        X : ndarray, shape (n_epochs, n_channels, n_times)
            The data on which to estimate the CSP.
        Returns
        -------
        self : instance of TRCA
            Returns the modified instance.
        """
        
        enhanced = []
        for classINX in range(len(self._classes)):
            X_filtered = np.dot(self.filter[classINX][:,:self.n_components].T, X)
            X_filtered = X_filtered.transpose((1, 0, 2))
            X_filtered = np.stack(X_filtered[i].ravel() for i in range(X.shape[0]))
            enhanced.append(X_filtered)
        enhanced = np.concatenate(enhanced,axis=-1)

        return enhanced
        

class CSP(BaseCSP):
    
    def transform(self, X):
        self.filters_  = self.filters_.T
        
        enhanced = []
        for classINX in range(len(self._classes)):
            X_filtered = np.dot(self.filter[classINX][:,:self.n_components].T, X)
            X_filtered = X_filtered.transpose((1, 0, 2))
            X_filtered = np.stack(X_filtered[i].ravel() for i in range(X.shape[0]))
            enhanced.append(X_filtered)
        enhanced = np.concatenate(enhanced,axis=-1)

        return enhanced
    
    def fit(self, X, y):
        """Estimate the CSP decomposition on epochs.

        Parameters
        ----------
        X : ndarray, shape (n_epochs, n_channels, n_times)
            The data on which to estimate the CSP.
        y : array, shape (n_epochs,)
            The class for each epoch.

        Returns
        -------
        self : instance of CSP
            Returns the modified instance.
        """
        self._check_Xy(X, y)

        self._classes = np.unique(y)
        n_classes = len(self._classes)

        covs, sample_weights = self._compute_covariance_matrices(X, y)
        eigen_vectors, eigen_values = self._decompose_covs(covs,
                                                           sample_weights)
        
        filters = []
        for classINX in range(n_classes):
            ix = self._order_components(covs[classINX], sample_weights[classINX], eigen_vectors[classINX],
                                        eigen_values[classINX], self.component_order)

            eigen_vector = eigen_vectors[classINX][:,ix]
            filters.append(eigen_vector)

        self.filters_ = filters

        return self

    def transform(self,X):
        """
        Parameters
        ----------
        X : ndarray, shape (n_epochs, n_channels, n_times)
            The data on which to estimate the CSP.
        Returns
        -------
        self : instance of TRCA
            Returns the modified instance.
        """
        
        enhanced = []
        for classINX in range(len(self._classes)):
            X_filtered = np.dot(self.filters_[classINX][:,:self.n_components].T, X)
            X_filtered = X_filtered.transpose((1, 0, 2))
            X_filtered = np.stack(X_filtered[i].ravel() for i in range(X.shape[0]))
            enhanced.append(X_filtered)
        enhanced = np.concatenate(enhanced,axis=-1)
        return enhanced
        
    def _decompose_covs(self, covs, sample_weights):
        n_classes = len(covs)
        eigen_vectors = []
        eigen_values = []
        for classINX in range(n_classes):
            eigen_value, eigen_vector = la.eigh(covs[classINX], covs.sum(0))
            eigen_vectors.append(eigen_vector)
            eigen_values.append(eigen_value)
        return eigen_vectors, eigen_values

        

        
