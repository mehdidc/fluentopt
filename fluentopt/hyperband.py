import numpy as np

def hyperband(sample, run_batch, max_iter=81, eta=3, random_state=None):
    """
    Implementation of hyperband.
    Based on: <https://people.eecs.berkeley.edu/~kjamieson/hyperband.html>
    
    Parameters
    ----------

    sample :

    run_batch :
    
    max_iter :

    eta:

    random_state :
    """
    rng = np.random.RandomState(random_state)
    s_max = int(np.log(max_iter) / np.log(eta))
    B = (s_max + 1) * max_iter
    input_history_ = []
    output_history_ = []
    for s in reversed(range(s_max+1)):
        n = int(np.ceil(B / max_iter / (s+1) * eta**s))
        r = max_iter * eta**(-s)
        T = [sample(rng) for _ in range(n)] 
        for i in range(s+1):
            n_i = n * eta**(-i)
            r_i = r * eta**(i)
            keep = int(n_i/eta)
            values = run_batch([(r_i, t) for t in T])
            values = list(values)
            input_history_.extend([(r_i, t) for t in T])
            output_history_.extend(values)
            ind = np.argsort(values)
            values = [values[i] for i in ind][0:keep]
            T = [T[i] for i in ind][0:keep]
    return input_history_, output_history_
