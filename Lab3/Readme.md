# Лабораторная работа №3

## Графики Optuna

Графики были получены с помощью Optuna dashboard. 

Для подключения к дашборду нужно ввести:

```
optuna-dashboard postgresql://optuna_user:12345@localhost/optuna_db
```

### TPE sampler + Hyperband pruner

![image info](./img/TPE1.jpg)

![image info](./img/TPE2.jpg)

### CMAeS sampler + Wilcoxon pruner

![image info](./img/CMAES1.jpg)

![image info](./img/CMAES2.jpg)