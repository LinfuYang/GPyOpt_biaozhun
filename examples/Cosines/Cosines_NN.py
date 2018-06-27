import GPy
import GPyOpt
from GPyOpt.acquisitions.LCB import AcquisitionLCB
from numpy.random import seed

seed(123)
func = GPyOpt.objective_examples.experiments2d.cosines()
print('x_min:', func.min, end=' ')
print('fx_min:', func.fmin)
func.plot()

objective = GPyOpt.core.task.SingleObjective(func.f)

space = GPyOpt.Design_space(space=[{'name': 'var_1', 'type': 'continuous', 'domain': (0, 1)},
                                   {'name': 'var_2', 'type': 'continuous', 'domain': (0, 1)}])


# 聚类时采点的个数
mun_point = 150
# k  表示簇个数
for k in range(5, 6):
    # 模型
    model = GPyOpt.models.GPModel(optimize_restarts=10, verbose=False)
    # 优化函数（优化AF）
    acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(space, optimizer='GMM_NN', cluster_k=k,
                                                                     point_num=mun_point)
    # 初始化采点
    initial_design = GPyOpt.experiment_design.initial_design('random', space, 5)
    # print('initial_design:', initial_design)

    # 实例化AF函数
    acquisition = AcquisitionLCB(model, space, optimizer=acquisition_optimizer)

    # 根据已知条件给出最理想的下一个x*
    evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

    # 实例化贝叶斯模型
    # noemalize_y = False 表示先验均值为零、、TRUE表示 为训练数据的均值
    bo = GPyOpt.methods.ModularBayesianOptimization(model, space, objective, acquisition, evaluator, initial_design, normalize_Y=False)

    # 最大迭代次数
    max_iter = 200

    # 命名规则
    name = 'Cosines_NN_%s_' % k + str(mun_point)

    # 执行贝叶斯优化
    bo.run_optimization(max_iter=max_iter, evaluations_file=name)

    # 画图
    bo.plot_convergence(filename=name, func_name=name)















