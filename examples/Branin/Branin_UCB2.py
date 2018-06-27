import GPyOpt
from GPyOpt.acquisitions.LCB import AcquisitionLCB
from numpy.random import seed
seed(123)

func = GPyOpt.objective_examples.experiments2d.branin()
print('x_min:', func.min, end=' ')
print('fx_min:', func.fmin)
# func.plot()

objective = GPyOpt.core.task.SingleObjective(func.f)

space = GPyOpt.Design_space(space=[{'name': 'var_1', 'type': 'continuous', 'domain': (-5, 10)},
                                   {'name': 'var_2', 'type': 'continuous', 'domain': (1, 15)}])

mun_point = 150
for k in range(10, 11):
    model = GPyOpt.models.GPModel(optimize_restarts=5, verbose=False)

    acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(space, optimizer='GMM_UCB2', cluster_k=k,
                                                                     point_num=mun_point)

    initial_design = GPyOpt.experiment_design.initial_design('random', space, 5)
    # print('initial_design:', initial_design)

    acquisition = AcquisitionLCB(model, space, optimizer=acquisition_optimizer)

    evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

    bo = GPyOpt.methods.ModularBayesianOptimization(model, space, objective, acquisition, evaluator, initial_design,
                                                    normalize_Y=True)

    max_iter = 200
    name = 'Branin_UCB2_%s_' % k + str(mun_point)
    bo.run_optimization(max_iter=max_iter, evaluations_file=name)

    bo.plot_convergence(filename=name, func_name=name)