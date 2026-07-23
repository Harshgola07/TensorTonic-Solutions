import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) :
    d_k = Q.shape[-1]

    scores = Q @ K.transpose(0, 2, 1)
    scores = softmax(scores / (d_k ** 0.5))

    return scores @ V
    

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    Q_projected = Q @ W_q
    K_projected = K @ W_k
    V_projected = V @ W_v

    batch_size, seq_len, d_model = Q.shape
    result = np.empty((batch_size, seq_len, 0))
    d_k = d_model // num_heads
    
    for i in range(0, d_model, d_k) :
        Q_i = Q_projected[:, :, i:i+d_k]
        K_i = K_projected[:, :, i:i+d_k]
        V_i = V_projected[:, :, i:i+d_k]

        head = attention(Q_i, K_i, V_i)
        result = np.concatenate((result, head), axis=2)

    result = result @ W_o
    return result
        