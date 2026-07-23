import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)

    normalized = (gamma * (x - mean)) / np.sqrt(variance + eps) + beta
    return normalized

def self_attention(Q: np.ndarray, K: np.ndarray, V:np.ndarray) :
    d_k = Q.shape[-1]

    scores = Q @ K.transpose(0, 2, 1)
    scores = softmax(scores / np.sqrt(d_k)) @ V

    return scores
    
def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
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

        head = self_attention(Q_i, K_i, V_i)
        result = np.concatenate((result, head), axis=2)

    result = result @ W_o
    return result

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    return np.maximum(0, x @ W1 + b1) @ W2 + b2

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    multi_head_output = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    x = layer_norm(x + multi_head_output, gamma1, beta1)
    output = layer_norm(x + feed_forward(x, W1, b1, W2, b2), gamma2, beta2)

    return output
    