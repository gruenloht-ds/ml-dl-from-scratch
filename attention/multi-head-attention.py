import numpy as np

class Attention:
    def __init__(self, d_model, h, p_dropout):
        self.v = np.random.randn(d_model, d_model)
        self.q = np.random.randn(d_model, d_model) 
        self.k = np.random.randn(d_model, d_model) 
        self.o = np.random.randn(d_model, d_model)
        
        self.d_model = d_model
        self.h = h 
        self.d_k = d_model // h
        self.p_dropout = p_dropout

    def softmax(self, x, axis=None):
        x_shifted = x - x.max(axis=axis, keepdims=True)  # for numerical stability
        return np.exp(x_shifted) / np.exp(x_shifted).sum(axis=axis, keepdims=True)

    def multiply_qkv(self, input_data, x):
        # Do matrix multiplication
        X = input_data @ x                                             # [batch_size, seq_len, d_model]

        # Split the multiplication into 8 heads (blockwise mat mult)
        X = X.reshape(X.shape[0], X.shape[1], self.h, -1)              # [batch_size, seq_len, heads, d_k]

        # Move the heads next to the batches so we can stack them
        X = X.transpose(0, 2, 1, 3)                                    # [batch_size, heads, seq_len, d_k]
        
        # Combine batches with the attention heads
        X = X.reshape(-1, X.shape[2], X.shape[3])                      # [batch_size * heads, seq_len, d_k]
        
        return X

    def concat(self, X):
        # Reverse the previous steps from multiply_qkv
        X = X.reshape(-1, self.h, X.shape[1], X.shape[2])         # [batch_size, heads, seq_len, d_k]
        X = X.transpose(0, 2, 1, 3)                               # [batch_size, seq_len, heads, d_k]
        X = X.reshape(X.shape[0], X.shape[1], -1)                 # [batch_size, seq_len, d_model]
        return X

    def dropout(self, mat):
        masking = np.random.binomial(1, 1-self.p_dropout, mat.shape)/1.0
        masking /= (1-self.p_dropout)

        return mat * masking

    def forward(self, X, attention_mask=None):
        # X: (batch_size, seq_len, d_k)

        # Linear projections
        Q = self.multiply_qkv(X, self.q) # [batch_size * h, seq_len, d_k]
        K = self.multiply_qkv(X, self.k) # [batch_size * h, seq_len, d_k]
        V = self.multiply_qkv(X, self.v) # [batch_size * h, seq_len, d_k]


        # Scaled dot product
        scores = Q @ K.transpose(0,2,1) / Q.shape[-1] ** 0.5  # [batch_size * h, seq_len, seq_len]

        # Apply masking
        if attention_mask is not None:
            attention_mask *= -1e9 # masking is assumed to be a matrix of 1's and 0's
            scores += attention_mask
        
        # Attention weights
        self.attention = self.softmax(scores, axis=-1)  # [batch_size * h, seq_len, seq_len]

        attention_dropout = self.dropout(self.attention)

        # Apply attention to values
        multi_head_attention = attention_dropout @ V  # [batch_size * h, seq_len, d_k]

        # Concatenate heads (reverse previous manipulation)        
        multi_head_attention = self.concat(multi_head_attention)

        output = (multi_head_attention @ self.o)
        
        return output

    def __call__(self, X, attention_mask=None):
        return self.forward(X, attention_mask)
