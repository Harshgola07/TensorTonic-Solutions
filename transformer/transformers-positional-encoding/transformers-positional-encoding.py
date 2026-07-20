import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    PE = []
    for pos in range(seq_length) :
        row = []
        for i in range(d_model) :
            denominator = np.power(10000, 2 * (i // 2) / d_model)
            if i & 1 :
                row.append(np.cos(pos / denominator))
            else :
                row.append(np.sin(pos / denominator))

        PE.append(row)

    return np.array(PE)
                