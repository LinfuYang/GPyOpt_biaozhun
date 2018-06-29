# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import GPyOpt
from numpy.random import seed
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist
import pylab as pl


class Optimizer(object):
    """
    Class for a general acquisition optimizer.

    :param bounds: list of tuple with bounds of the optimizer
    """

    def __init__(self, bounds):
        self.bounds = bounds

    def optimize(self, x0, f=None, df=None, f_df=None):
        """
        :param x0: initial point for a local optimizer.
        :param f: function to optimize.
        :param df: gradient of the function to optimize.
        :param f_df: returns both the function to optimize and its gradient.
        """
        raise NotImplementedError("The optimize method is not implemented in the parent class.")


class OptGMM_NN(Optimizer):
    '''
        Wrapper for GMM to use the true or the approximate gradients.
    '''
    def __init__(self, space=None, bounds=None):
        self.space = space
        super(OptGMM_NN, self).__init__(bounds)

    def labelslable(self, dataSet, center, labels, k):
        m = np.shape(dataSet)[0]
        cluster_splt = np.zeros((m, 2))
        for j in range(m):
            for i in range(k):
                #
                if labels[j] == i:
                    cluster_splt[j, 0] = i
                    x = np.array([dataSet[j]])
                    mu = np.array([center[i]])
                    # print('X:', x)
                    # print('mu:', mu)
                    cluster_splt[j, 1] = (self.l2norm_(x, mu))
        return cluster_splt

    def l2norm_(self, X, Xstar):

        return cdist(X, Xstar)

    # 由于本函数需要进行优化多中函数，所以不能改变其参数，对于需要添加的参数可以存放在构造函数中
    def optimize(self, tempt=None, f=None, k=5, point_num=200):

        # print('self.temp_t:', self.temp_t)
        # 生成需要聚类所需的样本点
        initial_design = GPyOpt.experiment_design.initial_design('random', self.space, point_num)
        # print('initial_design:', np.shape(initial_design)[0])
        # print('initial_design:', initial_design[0])

        # test_vale, sigma = f(np.array([69.64691856]))
        # print('test_value:', test_vale)
        # print('sigma:', sigma)
        m, n = np.shape(initial_design)
        # 用于保存每个矩阵的均值和方差
        dataSet = np.zeros((m, 2))
        # 计算每个样本的均值和方差
        for index, point in enumerate(initial_design):
            # print('point:', point)
            dataSet[index, 0], dataSet[index, 1] = f(point)

        # print('dataSet:', dataSet)
        gmm = GaussianMixture(n_components=k, covariance_type='full', init_params='kmeans', max_iter=150)
        gmm.fit(dataSet)
        # 聚类中心
        centroids = gmm.means_
        # print('centroids:', centroids)
        # 没个样本所属的类别
        labels = gmm.predict(dataSet)
        # print('labels:', labels)
        # 存放类别和该样本到簇中心的距离
        clusterAssign_GMM = self.labelslable(dataSet, centroids, labels, k)

        # colValue = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
        # for c in range(k):
        #     cor_X = []
        #     cor_Y = []
        #     for a in range(m):
        #         if clusterAssign_GMM[a, 0] == c:
        #             cor_X.append(dataSet[a, 0])
        #             cor_Y.append(dataSet[a, 1])
        #     pl.scatter(cor_X, cor_Y, marker='x', color=colValue[c % len(colValue)], label=c)
        # pl.legend(loc='upper right')
        # pl.show()
        # 对簇中元素进行去重,并进行排序
        label_unique = np.unique(labels)
        # label_unique 中的元素表示，聚类后，存在元素的簇有哪些
        UCB_centro = []
        for i in label_unique:
            # 将label_unique 中元素取出
            UCB_centro.append(-centroids[i, 0] + tempt * centroids[i, 1])
        # 找UCB最大的簇心编号
        max_signal = np.argmax(UCB_centro)
        centro_center = label_unique[max_signal]
        min_distance = np.inf
        best_x_value = -1
        # 找最优簇中离簇中心最近的点
        for j in range(m):
            if clusterAssign_GMM[j, 0] == centro_center:
                if min_distance > clusterAssign_GMM[j, 1]:
                    min_distance = clusterAssign_GMM[j, 1]
                    best_x_value = initial_design[j]




        result_x = np.atleast_2d(best_x_value)
        return result_x



class OptGMM_UCB2(Optimizer):
    '''
        Wrapper for GMM to use the true or the approximate gradients.
    '''
    def __init__(self, space=None, bounds=None):
        self.space = space
        super(OptGMM_UCB2, self).__init__(bounds)
    def labelslable(self, dataSet, center, labels, k):
        m = np.shape(dataSet)[0]
        cluster_splt = np.zeros((m, 2))
        for j in range(m):
            for i in range(k):
                #
                if i == labels[j]:
                    cluster_splt[j, 0] = i
                    x = np.mat(dataSet[j])
                    mu = np.mat(center[i])
                    cluster_splt[j, 1] = (self.l2norm_(x, mu))
        return cluster_splt
    def l2norm_(self, X, Xstar):

        return cdist(X, Xstar)

    # 由于本函数需要进行优化多中函数，所以不能改变其参数，对于需要添加的参数可以存放在构造函数中
    def optimize(self, tempt=None, f=None, k=10, point_num=200):

        # 生成需要聚类所需的样本点
        initial_design = GPyOpt.experiment_design.initial_design('random', self.space, point_num)

        # 用于保存每个矩阵的均值和方差
        dataSet = np.zeros((point_num, 2))

        # 计算每个样本的均值和方差
        for index, point in enumerate(initial_design):
            dataSet[index, 0], dataSet[index, 1] = f(point)

        gmm = GaussianMixture(n_components=k, covariance_type='full', init_params='kmeans', max_iter=150)
        gmm.fit(dataSet)

        # 聚类中心
        centroids = gmm.means_

        # 每个样本所属的类别
        labels = gmm.predict(dataSet)

        # 存放类别和该样本到簇中心的距离
        clusterAssign_GMM = np.mat(self.labelslable(dataSet, centroids, labels, k))

        # 对簇中元素进行去重,并进行排序
        label_unique = np.unique(labels)

        # label_unique 中的元素表示，聚类后，存在元素的簇有哪些
        UCB_centro = []
        for i in label_unique:
            # 将label_unique 中元素取出
            UCB_centro.append(-centroids[i, 0] + tempt * centroids[i, 1])

        # 找UCB最大的下标
        max_signal = np.argmax(UCB_centro)

        # 找UCB最大的簇心编号
        centro_center = label_unique[max_signal]

        '''
        找最优簇中对UCB进行优化
        由于clusterAssing_GMM返回的是每个样本所属的簇和 到簇中心的距离
        我们需要:1.需要最优簇中所有数据的均值和方差，2.需要知道簇中样本值数组ptsInClust 保存最优簇中所有元素的均值和方差
        '''

        # 只是为了找出最优簇中的元素个数，并不利用ptsInClust中的元素
        ptsInClust = dataSet[np.nonzero(clusterAssign_GMM[:, 0].A == centro_center)[0]]
        # 最优簇中样本个数
        m_centro = np.shape(ptsInClust)[0]



        # 样本的维度
        n = np.shape(initial_design)[1]

        # 定义数组存放 最优簇中的元素
        cluster_dataSet = np.zeros((m_centro, n))

        # 找最优簇中离簇中心最近的点
        # 用于存放最优簇中的UCB值
        UCB_opt_value = np.zeros((m_centro))
        for cent in range(m_centro):
            for flage in range(point_num):
                # 如果 clusterAssign_GMM 中关于样本的簇标等于最优簇的簇标
                if clusterAssign_GMM[flage, 0] == centro_center:
                    # 就把对应样本保存下来
                    cluster_dataSet[cent] = initial_design[flage]

            UCB_opt_value[cent] = -ptsInClust[cent, 0] + tempt * ptsInClust[cent, 1]

        index_opt = np.argmax(UCB_opt_value)
        # 最优点的UCB
        best_x_value = cluster_dataSet[index_opt]
        result_x = np.atleast_2d(best_x_value)
        return result_x




class OptLbfgs(Optimizer):
    '''
    Wrapper for l-bfgs-b to use the true or the approximate gradients.
    '''
    def __init__(self, bounds, maxiter=1000):
        super(OptLbfgs, self).__init__(bounds)
        self.maxiter = maxiter

    def optimize(self, x0, f=None, df=None, f_df=None):
        """
        :param x0: initial point for a local optimizer.
        :param f: function to optimize.
        :param df: gradient of the function to optimize.
        :param f_df: returns both the function to optimize and its gradient.
        """
        import scipy.optimize
        if f_df is None and df is not None: f_df = lambda x: float(f(x)), df(x)
        if f_df is not None:
            def _f_df(x):
                return f(x), f_df(x)[1][0]
        if f_df is None and df is None:
            res = scipy.optimize.fmin_l_bfgs_b(f, x0=x0, bounds=self.bounds,approx_grad=True, maxiter=self.maxiter)
        else:
            res = scipy.optimize.fmin_l_bfgs_b(_f_df, x0=x0, bounds=self.bounds, maxiter=self.maxiter)

        ### --- We check here if the the optimizer moved. It it didn't we report x0 and f(x0) as scipy can return NaNs
        if res[2]['task'] == b'ABNORMAL_TERMINATION_IN_LNSRCH':
            result_x  = np.atleast_2d(x0)
            result_fx =  np.atleast_2d(f(x0))
        else:
            result_x = np.atleast_2d(res[0])
            result_fx = np.atleast_2d(res[1])

        return result_x, result_fx


class OptDirect(Optimizer):
    '''
    Wrapper for DIRECT optimization method. It works partitioning iteratively the domain
    of the function. Only requires f and the box constraints to work.

    '''
    def __init__(self, bounds, maxiter=1000):
        super(OptDirect, self).__init__(bounds)
        self.maxiter = maxiter
        assert self.space.has_types['continuous']

    def optimize(self, x0, f=None, df=None, f_df=None):
        """
        :param x0: initial point for a local optimizer.
        :param f: function to optimize.
        :param df: gradient of the function to optimize.
        :param f_df: returns both the function to optimize and its gradient.
        """
        # Based on the documentation of DIRECT, it does not seem we can pass through an initial point x0
        try:
            from DIRECT import solve
            def DIRECT_f_wrapper(f):
                def g(x, user_data):
                    return f(np.array([x])), 0
                return g
            lB = np.asarray(self.bounds)[:,0]
            uB = np.asarray(self.bounds)[:,1]
            x,_,_ = solve(DIRECT_f_wrapper(f),lB,uB, maxT=self.maxiter)
            return np.atleast_2d(x), f(np.atleast_2d(x))
        except ImportError:
            print("Cannot find DIRECT library, please install it to use this option.")


class OptCma(Optimizer):
    '''
    Wrapper the Covariance Matrix Adaptation Evolutionary strategy (CMA-ES) optimization method. It works generating
    an stochastic search based on multivariate Gaussian samples. Only requires f and the box constraints to work.

    '''
    def __init__(self, bounds, maxiter=1000):
        super(OptCma, self).__init__(bounds)
        self.maxiter = maxiter

    def optimize(self, x0, f=None, df=None, f_df=None):
        """
        :param x0: initial point for a local optimizer.
        :param f: function to optimize.
        :param df: gradient of the function to optimize.
        :param f_df: returns both the function to optimize and its gradient.
        """
        try:
            import cma
            def CMA_f_wrapper(f):
                def g(x):
                    return f(np.array([x]))[0][0]
                return g
            lB = np.asarray(self.bounds)[:,0]
            uB = np.asarray(self.bounds)[:,1]
            x = cma.fmin(CMA_f_wrapper(f), x0, 0.6, options={"bounds":[lB, uB], "verbose":-1})[0]
            return np.atleast_2d(x), f(np.atleast_2d(x))
        except ImportError:
            print("Cannot find cma library, please install it to use this option.")
        except:
            print("CMA does not work in problems of dimension 1.")


def apply_optimizer(optimizer, x0, f=None, df=None, f_df=None, duplicate_manager=None, context_manager=None, space=None):
    """
    :param x0: initial point for a local optimizer (x0 can be defined with or without the context included).
    :param f: function to optimize.
    :param df: gradient of the function to optimize.
    :param f_df: returns both the function to optimize and its gradient.
    :param duplicate_manager: logic to check for duplicate (always operates in the full space, context included)
    :param context_manager: If provided, x0 (and the optimizer) operates in the space without the context
    :param space: GPyOpt class design space.
    """
    x0 = np.atleast_2d(x0)

    ## --- Compute a new objective that inputs non context variables but that takes into account the values of the context ones.
    ## --- It does nothing if no context is passed
    problem = OptimizationWithContext(x0=x0, f=f, df=df, f_df=f_df, context_manager=context_manager)


    if context_manager:
        add_context = lambda x : context_manager._expand_vector(x)
    else:
        add_context = lambda x : x

    if duplicate_manager and duplicate_manager.is_unzipped_x_duplicate(x0):
        raise ValueError("The starting point of the optimizer cannot be a duplicate.")

    ## --- Optimize point
    optimized_x, _ = optimizer.optimize(problem.x0_nocontext, problem.f_nocontext, problem.df_nocontext, problem.f_df_nocontext)

    ## --- Add context and round according to the type of variables of the design space
    suggested_x_with_context = add_context(optimized_x)
    suggested_x_with_context_rounded = space.round_optimum(suggested_x_with_context)

    ## --- Run duplicate_manager
    if duplicate_manager and duplicate_manager.is_unzipped_x_duplicate(suggested_x_with_context_rounded):
        suggested_x, suggested_fx = x0, np.atleast_2d(f(x0))
    else:
        suggested_x, suggested_fx = suggested_x_with_context_rounded, f(suggested_x_with_context_rounded)

    return suggested_x, suggested_fx


class OptimizationWithContext(object):

    def __init__(self, x0, f, df=None, f_df=None, context_manager=None):
        '''
        Constructor of an objective function that takes as input a vector x of the non context variables
        and retunrs a value in which the context variables have been fixed.
        '''
        self.x0 = np.atleast_2d(x0)
        self.f = f
        self.df = df
        self.f_df = f_df
        self.context_manager = context_manager

        if not context_manager:
            self.x0_nocontext = x0
            self.f_nocontext  =  self.f
            self.df_nocontext  =  self.df
            self.f_df_nocontext = self.f_df

        else:
            self.x0_nocontext = self.x0[:,self.context_manager.noncontext_index]
            self.f_nocontext  = self.f_nc
            if self.f_df is None:
                self.df_nocontext = None
                self.f_df_nocontext = None
            else:
                self.df_nocontext = self.df_nc
                self.f_df_nocontext  = self.f_df_nc

    def f_nc(self,x):
        '''
        Wrapper of *f*: takes an input x with size of the noncontext dimensions
        expands it and evaluates the entire function.
        '''
        x = np.atleast_2d(x)
        xx = self.context_manager._expand_vector(x)
        if x.shape[0] == 1:
            return self.f(xx)[0]
        else:
            return self.f(xx)

    def df_nc(self,x):
        '''
        Wrapper of the derivative of *f*: takes an input x with size of the not
        fixed dimensions expands it and evaluates the gradient of the entire function.
        '''
        x = np.atleast_2d(x)
        xx = self.context_manager._expand_vector(x)
        _, df_nocontext_xx = self.f_df(xx)
        df_nocontext_xx = df_nocontext_xx[:,np.array(self.context_manager.noncontext_index)]
        return df_nocontext_xx

    def f_df_nc(self,x):
        '''
        Wrapper of the derivative of *f*: takes an input x with size of the not
        fixed dimensions expands it and evaluates the gradient of the entire function.
        '''
        x = np.atleast_2d(x)
        xx = self.context_manager._expand_vector(x)
        f_nocontext_xx , df_nocontext_xx = self.f_df(xx)
        df_nocontext_xx = df_nocontext_xx[:,np.array(self.context_manager.noncontext_index)]
        return f_nocontext_xx, df_nocontext_xx


def choose_optimizer(space, optimizer_name, bound):
        """
        Selects the type of local optimizer
        """
        if optimizer_name == 'lbfgs':
            optimizer = OptLbfgs(bound)

        elif optimizer_name == 'DIRECT':
            optimizer = OptDirect(bound)

        elif optimizer_name == 'GMM_NN':
            optimizer = OptGMM_NN(space=space, bounds=bound)

        elif optimizer_name == 'GMM_UCB2':
            optimizer = OptGMM_UCB2(space=space, bounds=bound)

        elif optimizer_name == 'CMA':
            optimizer = OptCma(bound)
        else:
            raise InvalidVariableNameError('Invalid optimizer selected.')

        return optimizer
