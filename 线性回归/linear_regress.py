"""
线性回归模型
"""
import numpy as np

def gaussian_function(x, mu, sigma):
    """
    线性回归模型的基函数可以选用“高斯”基函数
    """
    return np.exp(-0.5 * ((x - mu)**2 / sigma**2))


def sigmoid_function(x, mu, sigma):
    """
    线性回归模型的基函数可以选用“Sigmoid”函数
    """
    a = (x - mu) / sigma
    return 1 / (1 + np.exp(-a))


def polynominal_function(x, degree):
    """
    线性回归模型的基函数可以选用“多项式”函数
    """
    return (x**degree)


def regularization(w, lambd, q):
    """
    定义正则化项， 其中w是模型参数，lambd是正则化参数，q是正则化项的值
    """
    return lambd/2 * 
