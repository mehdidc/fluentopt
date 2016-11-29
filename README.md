# fluentopt
A flexible hyper-parameter optimization library.
Most hyper-parameter optimization libraries impose three main restrictions :

- they control the optimization loop
- they force the points to be represented by vectors
- they priors are very restricted, e.g gaussian, uniform or discrete uniform

the goal of fluentopt is to provide  hyper-parameter optimization library where :

- the optimization loop is controlled by the user (but we will provide also helpers).
- the points can be represented by a python dictionary to express conditionals rather than just a vector. The dictionaries can also support strings, varying length lists and special objects like 'None'.
- the priors of hyper-parameters are not restricted to some pre-defined probability distributions. 
  Users will just provide   samplers as a python function, that is, a function that takes a seed and returns 
  a python dictionary.

