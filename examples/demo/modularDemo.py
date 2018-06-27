import GPyOpt
import GPy
from numpy.random import seed
from GPyOpt.acquisitions.LCB import AcquisitionLCB
seed(123)
func = GPyOpt.objective_examples.experiments2d.branin()
print(func.min)
print(func.fmin)
func.plot()

objective = GPyOpt.core.task.SingleObjective(func.f)

space = GPyOpt.Design_space(space =[{'name': 'var_1', 'type': 'continuous', 'domain': (-5,10)},
                                    {'name': 'var_2', 'type': 'continuous', 'domain': (1,15)}])

model = GPyOpt.models.GPModel(optimize_restarts=5, verbose=False)

aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(space)

initial_design = GPyOpt.experiment_design.initial_design('random', space, 5)

acquisition = AcquisitionLCB(model, space, optimizer=aquisition_optimizer)

evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

bo = GPyOpt.methods.ModularBayesianOptimization(model, space, objective, acquisition, evaluator, initial_design)

max_iter = 10
bo.run_optimization(max_iter=max_iter)
# bo.plot_acquisition()
bo.plot_convergence()


