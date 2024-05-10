import torch as T
import torch.nn.functional as F
from f_model import FModel_V1, FModel_V2
from collections import Counter
import numpy as np

def to_cuda(tensor):
    """Utility function to move tensor to GPU if available."""
    return None if tensor is None else tensor.cuda()

class DIFFUSE:
    def __init__(self, k, lmd=0.01, q=0.1, steps=300, with_warm_up=True):
        """Initialize the DIFFUSE clustering model.
        Args:
            k (int): Number of potential clusters (can be larger than the actual number of clusters)
            lmd (float): Lambda regularization hyperparameter for the cohesion loss.
            q (float): Quantile for estimating gamma.
            steps (int): Number of training steps.
            with_warm_up (bool): Whether to start with a warm-up phase focusing on fidelity loss.
        """
        self.k = k
        self.lmd = lmd
        self.q = q
        self.steps = steps
        self.with_warm_up = with_warm_up
        self.initialized = False
        
    # ---------------------------------------------
    def _initialize(self, X, X_ori=None):
        """Initialize model parameters and estimate gamma for similarity calculation."""
        self.initialized = True
        self.gamma = self._estimate_gamma(X, self.q)
        self.f = FModel_V1(self.k) if X_ori is None else FModel_V2(X_ori.shape[1], self.k)
        self.f = self.f.cuda()
        self.opt = T.optim.Adam(list(self.f.parameters()), lr=0.01)
        
        if self.with_warm_up: # Initially start by minimizing only the fidelity loss.
            saved_lmd, self.lmd = self.lmd, 0.
            self.fit(X, X_ori)
            self.lmd = saved_lmd
        
    # ---------------------------------------------
    def loss_func(self, X, A, R):
        """Compute the overall loss function including fidelity and cohesion losses."""
        fidelity_loss = F.mse_loss(X, R)
        if self.lmd == 0.: return fidelity_loss
        
        S = self._pairwise_similarity(X)
        D = self._pairwise_kl_divergence(A)
        cohesion_loss = (S * D).mean()
        
        return fidelity_loss + self.lmd * cohesion_loss
        
    # ---------------------------------------------
    def fit(self, X, X_ori=None):
        """Fit the model using provided data.
        Args:
            X (Tensor): Processed input data.
            X_ori (Tensor, optional): Original raw input data if available.
        """
        X, X_ori = to_cuda(X), to_cuda(X_ori)
        
        if not self.initialized:
            self._initialize(X, X_ori)
        
        for i in range(self.steps):
            indices = self._sample_batch(len(X))
            X_batch = X[indices]
            X_ori_batch = None if X_ori is None else X_ori[indices]
            
            A, R, C = self._compute_ARC(X_batch, X_ori_batch)
            loss = self.loss_func(X_batch, A, R)
            
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            
        return self
        
    # ---------------------------------------------
    def predict(self, X, X_ori=None):
        """Predict cluster assignments for the data.
        Args:
            X (Tensor): Processed input data.
            X_ori (Tensor, optional): Original raw input data if available.
        """
        X, X_ori = to_cuda(X), to_cuda(X_ori)
        A, R, C = self._compute_ARC(X, X_ori)
        preds = self._ignore_empty_clusters(A) # Optional, or directly: preds = A.argmax(dim=1)
        return preds
        
    # ---------------------------------------------
    def _compute_ARC(self, X, X_ori=None):
        """Compute assignment matrix, representatives, and reconstructed data."""
        W = self.f(X) if X_ori is None else self.f(X_ori)
        A = T.softmax(W, dim=1)
        C = T.mm(A.T, X) / A.sum(dim=0, keepdim=True).T
        R = T.mm(A, C)
        
        return A, R, C
        
    # ---------------------------------------------
    def _pairwise_similarity(self, X):
        """Calculate pairwise similarity between data points."""
        XX = T.cdist(X, X) ** 2
        return T.exp(-self.gamma * XX)
    
    # ---------------------------------------------
    def _sample_batch(self, n, size=200):
        """Randomly sample batch indices for training."""
        indices = np.random.choice(n, size=size, replace=False) if size < n else range(n)
        return T.tensor(indices, dtype=T.long)
    
    # ---------------------------------------------
    def _estimate_gamma(self, X, q):
        """Estimate gamma for the exponential decay in similarity based on quantile."""
        indices = self._sample_batch(len(X), size=1000)
        X = X.cuda()[indices] # just for efficiency
        sqr_distances = T.cdist(X, X) ** 2
        gamma = 1. / (2. * T.quantile(sqr_distances, q))
        return gamma
    
    # ---------------------------------------------
    def _pairwise_kl_divergence(self, A, eps=1e-10):
        """Calculate pairwise KL divergence among pairs of data points."""
        log_A = T.log(A + eps)
        pairwise_log_diff = log_A.unsqueeze(1) - log_A.unsqueeze(0)
        kl_divergence = A.unsqueeze(1) * pairwise_log_diff
        kl_divergence = kl_divergence.sum(-1)
        return kl_divergence
    
    # ---------------------------------------------
    def _ignore_empty_clusters(self, A):
        """Ignore clusters that are empty or have just one point by reassigning their members."""
        preds = A.argmax(dim=1)
        
        # Ignoring very small or empty clusters
        cluster_sizes = Counter(preds.tolist())
        small_clusters = [cluster_id for cluster_id, size in cluster_sizes.items() if size <= 1]
        A[:, small_clusters] = 0.
        preds = A.argmax(dim=1)
        
        # Get unique cluster IDs and remap preds to contiguous range
        unique_ids = T.unique(preds, sorted=True)
        remapped_preds = T.searchsorted(unique_ids, preds)
        
        return remapped_preds.cpu()
        
