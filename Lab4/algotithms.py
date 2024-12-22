import numpy as np
import typing as tp
from scipy.stats import lognorm

class MetropolisHastings:
    '''
    Реализация метода сэмплирования Метрополиса-Гастингса
    '''
    def __init__(self, 
                 target_distribution: tp.Callable[[float], float],
                 proposal_distribution: tp.Callable[[float], float],
                 initial_state: float,
                 seed: int=42) -> None:
        self.targer_dist = target_distribution
        self.proposal_dist = proposal_distribution
        self.cur_state = initial_state
        self.samples = [initial_state]
        self.rng = np.random.default_rng(seed=seed)


    def _acceptance_probability(self, 
                                proposed_state: float) -> float:
        current_prob = self.targer_dist(self.cur_state)
        proposed_prob = self.proposal_dist(proposed_state)
        acceptance_ratio = proposed_prob / current_prob
        return acceptance_ratio


    def step(self) -> bool:
        proposal_state = self.proposal_dist(self.cur_state)
        acceptance_ratio = self._acceptance_probability(proposal_state)
        if self.rng.random() < acceptance_ratio:
            self.cur_state = proposal_state
            accepted = True
        else:
            accepted = False
        self.samples.append(self.cur_state)
        return accepted
    

    def run(self, num_steps) -> tp.List[float]:
        for _ in range(num_steps):
            self.step()
        return self.samples
    

class GibbsSampler:
    '''
    Реализация сэмплирования методом Гиббса
    '''
    def __init__(self, 
                 initial_state, 
                 data,
                 seed,):
        self.initial_state = initial_state
        self.data = data
        self.rng = np.random.default_rng(seed=seed)


    def run(self, num_steps):
        self.samples = np.zeros(num_steps)
        for i in range(1, num_steps):
            x = self.rng.choice([*self.data[:i], *self.data[i+1:]])
            self.samples[i] = x
        return self.samples