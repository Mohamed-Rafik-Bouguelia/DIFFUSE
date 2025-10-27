import torch as T

class FModel_V1(T.nn.Module):
    """A simple fully connected neural network model for generating clustering assignments."""
    
    def __init__(self, k):
        """Initializes a neural network with dense blocks and a final linear layer for cluster assignments.
        Args:
            k (int): Output dimension of the last layer.
        """
        super().__init__()
        self.dense_block1 = self.dense_block(20)
        self.dense_block2 = self.dense_block(20)
        self.dense_block3 = self.dense_block(20)
        self.last = T.nn.LazyLinear(k)
    
    def forward(self, X):
        X = self.dense_block1(X)
        X = self.dense_block2(X)
        X = self.dense_block3(X)
        return self.last(X)
    
    def dense_block(self, dim_out):
        return T.nn.Sequential(
            T.nn.LazyLinear(dim_out), 
            T.nn.BatchNorm1d(dim_out), 
            T.nn.ELU(),
        )

class FModel_V2(FModel_V1):
    """An extension of FModel_V1 that includes convolutional layers to handle raw data input."""
    
    def __init__(self, m, k):
        """Initializes a neural network that starts with convolutional layers followed by dense blocks.
        Args:
            m (int): Number of input channels for the first convolutional layer.
            k (int): Output dimension of the last linear layer.
        """
        super().__init__(k)
        self.conv_block1 = self.conv_block(m, 32)
        self.conv_block2 = self.conv_block(32, 16)
        self.flatten = T.nn.Flatten()
    
    def forward(self, X):
        X = self.conv_block1(X)
        X = self.conv_block2(X)
        X = self.flatten(X)
        X = self.dense_block1(X)
        X = self.dense_block2(X)
        X = self.dense_block3(X)
        return self.last(X)
    
    def conv_block(self, in_channels, out_channels):
        return T.nn.Sequential(
            T.nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1), 
            T.nn.BatchNorm1d(out_channels), 
            T.nn.ELU(),
        )
