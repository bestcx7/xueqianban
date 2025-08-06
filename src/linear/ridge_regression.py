import numpy as np

class RidgeRegression:
    """
    岭回归模型

    w* = argmin |t - X @ w| + alpha * |w|_2^2
    """

    def __init__(self, alpha: float = 1.):
        """
        初始化岭线形回归模型

        参数
        ----------
        alpha : float, optional
        """
        self.alpha = alpha

    def fit(self, x_trian: np.ndarray, y_train: np.ndarray):
        """
        最大化参数的后验估计

        参数
        ----------
        x_trian : np.ndarray
        y_train : np.ndarray
        """
        eye = np.eye(np.size(x_trian, 1)) # 创建单位矩阵
        self.w = np.linalg.solve(
            x_trian.T @ x_trian + self.alpha * eye,
            x_trian.T @ y_train
        ) # 求解 w 的线性方程组， w = (X^T * X + alpha * I)^-1 * X^T * y

    def predict(self, x: np.ndarray):
        """
        返回预测结果

        参数
        ----------
        x : np.ndarray

        返回
        ----------
        np.ndarray
        """

        return x @ self.w