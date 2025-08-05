import numpy as np

class LinearRegression:
    """
    线性回归模型

    y = X @ w
    t ~ N(t | X @ w, var)
    """
    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        """
        最小二乘法拟合

        参数
        —--------
        x_train : np.ndarray
        y_train : np.ndarray
        """
        self.w = np.linalg.pinv(x_train) @ y_train
        self.var = np.mean(np.square(x_train @ self.w - y_train))

    def predict(self, x: np.ndarray, return_std: bool = False):
        """
        返回给定输入的预测值

        参数
        ----------
        x : np.ndarray
        return_std : bool, optional

        返回
        ----------
        y : np.ndarray
        y_std : np.ndarray
        """
        y = x @ self.w
        if return_std:
            y_std = np.sqrt(self.var) * np.ones_like(y)
            return y, y_std
        return y    
