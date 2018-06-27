import GPy
import GPyOpt
from numpy.random import seed

f_true = GPyOpt.objective_examples.experiments1d.forrester()

bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (0, 1)}]

# f_true.plot()

seed(123)
myBopt = GPyOpt.methods.BayesianOptimization(f=f_true.f, domain=bounds, acquisition_type='EI', exact_feval=True)

max_iter = 15
max_time = 60
eps = 10e-8
myBopt.run_optimization(max_iter=max_iter, max_time=max_time, eps=eps)

# myBopt.plot_acquisition()
myBopt.plot_convergence()
