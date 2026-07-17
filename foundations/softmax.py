import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        maximum = np.max(z)

        intermediate = np.exp(z - maximum)
        sum_z = np.sum(intermediate)

        result = intermediate / sum_z
        return np.round(result, 4)
