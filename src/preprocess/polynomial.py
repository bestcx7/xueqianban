import numpy as np
import functools
from itertools import combinations_with_replacement

class PloynomialFeature:
    """
    多项式特征

    使用多项式特征转化输入数组

    示例
    =========
    x = [[a, b], [c, d]]

    y = PolynomialFeature(degree=2).transform(x)
    y = [[1, a, b, a^2, ab, b^2], [1, c, d, c^2, cd, d^2]]
    """

    def __init__(self, degree):
        """
        构造多项式特征

        参数
        ----------
        degree : int
            多项式的次数
        """
        assert isinstance(degree, int)
        self.degeree = degree

    def transform(self, x):
        """
        使用多项式特征转化输入数组

        参数
        ----------
        x : (sample_size, n) ndarray
            输入数组

        返回
        ----------
        output : (sample_size, 1 + nC1 + ... + nCd) ndarray
            多项式特征
        """
        if x.ndim == 1:
            x = x[:, None]
        x_t = x.transpose()
        features = [np.ones(len(x))]
        for degree in range(1, self.degeree + 1):
            for items in combinations_with_replacement(x_t, degree):
                features.append(functools.reduce(lambda x, y: x * y, items))
        return np.asarry(features).transpose()