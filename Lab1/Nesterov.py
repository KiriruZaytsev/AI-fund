import sympy
import numpy as np
import typing as tp

class Nesterov:
    '''
    Класс с реализацией модификации Nesterov momentum
    '''
    def __init__(self, 
                 x_bound: tp.Tuple[float, float],
                 y_bound: tp.Tuple[float, float],
                 lr: float=0.01, 
                 momentum: float=0.9, 
                 num_iter: float=100,
                 eps: float=1e-6,
                 seed: int=42):
        '''
        lr: Шаг градиентного спуска
        momentum: Коэффициент момента
        num_iter: Количество итераций
        seed: Начальное значение датчика случайной величины
        '''
        self.lr = lr
        self.momentum = momentum
        self.num_iter = num_iter
        self.rng = np.random.default_rng(seed=seed)
        self.x_bound = x_bound
        self.y_bound = y_bound
        self.eps = eps


    def optimize(self, 
                 func: tp.Callable[[np.array], float], 
                 gradient_func: tp.Optional[tp.Callable[[np.array], float]]=None) -> np.array:
        '''
        Оптимизация с использованием Nesterov Momentum
        Parameters:
            func: Функция для минимизации.
            gradient: Градиент функции.
            x_init: Начальная точка.
        Returns:
            Точка минимума после оптимизации.
        '''
        x_start = self.rng.uniform(low=[self.x_bound[0], self.y_bound[0]], 
                                   high=[self.x_bound[1], self.y_bound[1]])
        velocity = np.zeros_like(x_start)  
        for _ in range(self.num_iter):
            x_tilde = x_start + self.momentum * velocity
            if gradient_func is None:
                grad = self._calc_gradient(x_tilde, 
                                           func)
            else:
                grad = gradient_func(x_tilde)
            velocity = self.momentum * velocity - self.lr * grad
            x_new = x_start + velocity
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