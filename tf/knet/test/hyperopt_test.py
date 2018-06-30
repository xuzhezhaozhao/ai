
import hyperopt as hp


def f(x):
    x = x['x']
    print(x)
    return x ** 2


best = hp.fmin(fn=f,
               space={'x': hp.hp.choice('x', [2, 1, 3])},
               algo=hp.rand.suggest,
               max_evals=3)
print best
