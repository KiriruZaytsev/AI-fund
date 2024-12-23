import optuna
import numpy as np
import typing as tp
from functools import partial
from lightgbm import LGBMClassifier

postgresql_url = 'postgresql://optuna_user:12345@localhost/optuna_db'

def objective(trial, X_train, y_train, X_test, y_test, metric_func) -> float:
    params = {'num_leaves': trial.suggest_int('num_leaves', 2, 32),
              'max_depth': trial.suggest_int('max_depth', 2, 10),
              'learning_rate': trial.suggest_float('learning_rate', 0.01, 1),
              'n_estimators': trial.suggest_int('n_estimators', 50, 200)}
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return metric_func(y_test, y_pred)


def tune_hyperparameters(study_name: str,
                         X: tp.Tuple[tp.Any, tp.Any],
                         y: tp.Tuple[tp.Any, tp.Any],
                         storage: str,
                         sampler: tp.Any,
                         pruner: tp.Any,
                         metric_func: tp.Callable[[tp.Any, tp.Any], float],
                         direction: str='maximize',
                         n_trials: int=100,
                         timeout: int=5000) -> tp.Dict[str, tp.Any]:
    '''
    Функция для подбора гиперпараметров модели
    Parameters:
        model: Обёртка модели, должна иметь методы fit и predict
        X: Tuple с матрицей признаков обучающей и тестовой выборок 
        y: Tuple с таргетом обучающей и тестовой выборок 
        storage: База данных для общей памяти Optuna
        metric_func: Метрика измерения качества
        param_dict: Словарь с гиперпараметрами модели
        n_trial: Число триалов
        timeout: Таймаут подбора гиперпараметров
    Return:
        Словарь с подобранными гиперпараметрами модели
    '''
    X_train, X_test = X
    y_train, y_test = y
    study = optuna.create_study(study_name=study_name,
                                sampler=sampler,
                                pruner=pruner,
                                storage=storage,
                                load_if_exists=True,
                                direction=direction)        
    objective_partial = partial(objective, 
                                X_train=X_train, 
                                y_train=y_train, 
                                X_test=X_test, 
                                y_test=y_test, 
                                metric_func=metric_func)
    study.optimize(objective_partial, n_trials=n_trials, timeout=timeout)
    return study.best_params