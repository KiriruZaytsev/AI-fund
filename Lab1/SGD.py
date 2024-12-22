import sympy
import numpy as np
import typing as tp

class SGD:
    '''
    Класс с реализацией стохастического градиентного спуска
    '''
    def __init__(self,
                 x_bound: tp.Tuple[float, float],
                 y_bound: tp.Tuple[float, float],
                 num_iter: int=100,
                 lr: tp.Optional[float]=None,
                 use_symbolic_gradient=False,
                 eps: float=1e-6,
                 seed: int=42):
        '''
        x_bound: Границы переменной x
        y_bound: Границы переменной y
        num_iter: Количество итераций градиентного спуска
        lr: Опциональный параметр шага градиентного спуска. Если None, то в качестве
            шага будет взята последовательность 1/n, где n - номер итерации алгоритма
        use_symbolic_gradient: Использовать символьное вычисление градиента
        eps: Значение малой эпсилон 
        seed: Начальное значение датчика случайной величины
        '''
        self.lr = lr
        self.num_iter = num_iter
        self.symbolic = use_symbolic_gradient
        self.rng = np.random.default_rng(seed=seed)
        self.eps = eps
        self.x_bound = x_bound
        self.y_bound = y_bound


    def optimize(self, 
                 func: tp.Callable[[float, float], float],
                 gradient: tp.Optional[tp.Callable[[float, float], float]]=None) -> np.array:
        '''
        Нахождение оптимума с помощью градиентного спуска
        Parameters:
            func: Оптимизируемая функция
            gradient: Опциональный параметр - градиент функции. Если None, то для вычисления
                градиента будет использована центральная разностная схема
        Return:
            Точка экстремума функции
        '''
        x_start = self.rng.uniform(low=[self.x_bound[0], self.y_bound[0]], 
                                   high=[self.x_bound[1], self.y_bound[1]])
        for epoch in range(1, self.num_iter + 1):
            if gradient is None:
                grad = self._calc_gradient(x_start, 
                                           func)
                if self.lr is None:
                    x_new = x_start - (1 / epoch) * grad
                else:
                    x_new = x_start - self.lr * grad
            else:
                if self.lr is None:
                    x_new = x_start - (1 / epoch) * gradient(x_start)
                else:
                    x_new = x_start - self.lr * gradient(x_start)
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
    