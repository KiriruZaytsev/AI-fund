import sympy
import numpy as np
import typing as tp

class Adam:
    '''
    Класс с реализацией метода Adam
    '''
    def __init__(self,
                 x_bound: tp.Tuple[float, float],
                 y_bound: tp.Tuple[float, float],
                 lr: float=0.01,
                 beta_1: float=0.9,
                 beta_2: float=0.999,
                 num_iter: int=100,
                 eps: float=1e-6,
                 seed: int=42):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.num_iter = num_iter
        self.x_bound = x_bound
        self.y_bound = y_bound
        self.eps = eps
        self.bias = 0
        self.var = 0
        self.iter = 0
        self.rng = np.random.default_rng(seed=seed)


    def step(self, 
             grad: np.array) -> np.array:
        self.iter += 1
        self.bias = self.beta_1 * self.bias + (1 - self.beta_1) * grad
        self.var = self.beta_2 * self.var + (1 - self.beta_2) * np.square(grad)
        bias_hat = self.bias / (1 - self.beta_1 ** self.iter)
        var_hat = self.var / (1 - self.beta_2 ** self.iter)
        update = self.lr * bias_hat / (np.sqrt(var_hat) + self.eps)
        return update

    def optimize(self,
                 func: tp.Callable[[np.array], float],
                 gradient_func: tp.Optional[tp.Callable[[np.array], float]]=None) -> np.array:
        x_start = self.rng.uniform(low=[self.x_bound[0], self.y_bound[0]], 
                                   high=[self.x_bound[1], self.y_bound[1]])
        for _ in range(self.num_iter):
            if gradient_func is None:
                grad = self._calc_gradient(x_start, func)
            else:
                grad = gradient_func(x_start)
            update = self.step(grad)
            x_new = x_start - update
            x_start = x_new
        return x_start

    
    def _calc_gradient(self,
                       x: np.array, 
                       func: tp.Callable[[np.array], float]) -> np.array:
        '''
        Численное вычисление градиента
        Parameters:
            x: Точка, в которой нужно посчитать градиент
            func: Оптимизируемая функция
        Return:
            Вектор градиента в точке x
        '''
        grad = np.zeros_like(x)
        for i in range(len(x)):
            point_forward = x.copy()
            point_backward = x.copy()
            point_forward[i] += self.eps
            point_backward[i] -= self.eps
            grad[i] = (func(point_forward) - func(point_backward)) / (2 * self.eps)
        return grad