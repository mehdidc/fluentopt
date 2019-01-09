"""
======================================================
Hyperband
======================================================

An example with usage of hyperband.
"""

import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_boston

from fluentopt.hyperband import hyperband

if __name__ == "__main__":
    np.random.seed(42)
    data = load_boston()
    X = data["data"]
    y = data["target"]
    ind = np.arange(len(X))
    np.random.shuffle(ind)
    X = X[ind]
    y = y[ind]
    X_train = X[0:400]
    y_train = y[0:400]
    X_test = X[400:]
    y_test = y[400:]

    def sample(rng):
        return {"max_depth": rng.randint(1, 10), "learning_rate": rng.uniform(0, 1)}

    def run_batch(batch):
        for num_iters, params in batch:
            max_depth = params["max_depth"]
            learning_rate = params["learning_rate"]
            num_iters = int(num_iters)
            reg = GradientBoostingRegressor(
                learning_rate=learning_rate, max_depth=max_depth, n_estimators=num_iters
            )
            reg.fit(X_train, y_train)
            mse = ((reg.predict(X_test) - y_test) ** 2).mean()
            yield mse

    input_hist, output_hist = hyperband(
        sample, run_batch, max_iter=100, random_state=42
    )
    idx = np.argmin(output_hist)
    nb_iter, params = input_hist[idx]
    mse = output_hist[idx]
    print("Total nb. evaluations : {}".format(len(input_hist)))
    print("Best nb. of iterations : {}".format(int(nb_iter)))
    print("Best params : {}".format(params))
    print("Best mse : {:.3f}".format(mse))
    r2 = 1.0 - mse / y_test.var()
    print("R2 : {:.3f}".format(r2))
