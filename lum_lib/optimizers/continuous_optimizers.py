import scipy.optimize as spo
import numpy as np
import copy
import os
import pickle
import matplotlib.pyplot as plt
import time

from lum_lib.utils.plotter import Plotter

class Adam_Optimizer(object):
    """
    use adam to continuously optimize the design region
    """

    def __init__(self, optimization_problem, step_num_list, step_size=1e-2,
                 bounds=None, direction='min', beta1=0.9, beta2=0.999, device_name='device'):
        self.optimization_problem = optimization_problem
        self.objective_func = self.optimization_problem.objective_function

        self.beta1 = beta1
        self.beta2 = beta2
        self.direction = direction
        self.bounds = bounds
        self.step_size = step_size
        self.device_name = device_name

        # create device results folder
        self.save_folder = os.path.join(optimization_problem.workingDir, 'results')
        if not os.path.exists(self.save_folder):
            os.mkdir(self.save_folder)

        self.plotter = Plotter(save_folder=self.save_folder)

        self.step_num_list = step_num_list

        self.log_file = os.path.join(self.save_folder, self.device_name + '_c_log.txt')

    def run_optimization_stage(self):
        self.total_stage = len(self.step_num_list)

        with open(self.log_file, 'a') as logs:
            logs.write('step\tobjective_value\ttrans_coefficient\n')             #记录优化过程中的目标值和透射系数

        # initialize ADAM parameters to be None
        mopt = None
        vopt = None

        for i in range(self.total_stage):
            self.current_stage = i + 1
            start_step_num = 0 if i == 0 else self.step_num_list[i - 1]
            stop_step_num = self.step_num_list[i]

            print('current optimization stage: {}'.format(self.current_stage))

            params, grad_adam, mopt, vopt = self.run_optimization_step(start_step=start_step_num,
                                                                       stop_step=stop_step_num, mopt=mopt,
                                                                       vopt=vopt)
            self.optimization_problem.geometry.update_projection()


    def run_optimization_step(self, start_step, stop_step, mopt=None, vopt=None):

        # the parameters to be updated, the rho_vector (normalized to 0-1)
        params = self.optimization_problem.geometry.get_current_params()

        for iteration in range(start_step, stop_step):                 #从start_step到stop_step进行迭代

            t_start = time.time()

            self.optimization_problem.initialize()
            g_adjoint = self.optimization_problem.callable_jac(params)                        #算出目标函数关于参数的梯度

            fom = self.optimization_problem.current_fom
            transmission_coef = self.optimization_problem.fom_transmission_coeff_list

            t_elapsed = time.time() - t_start

            self.print_step(mode_objective_value=fom,
                            transmission_coef=transmission_coef,
                            time_used=t_elapsed,
                            current_step=iteration - start_step,
                            all_steps=stop_step - start_step)

            if mopt is None and vopt is None:
                mopt = np.zeros(g_adjoint.shape)
                vopt = np.zeros(g_adjoint.shape)

            (grad_adam, mopt, vopt) = self._step_adam(g_adjoint, mopt, vopt, iteration, self.beta1, self.beta2)

            if self.direction == 'min':
                params = params - self.step_size * grad_adam
            elif self.direction == 'max':
                params = params + self.step_size * grad_adam
            else:
                raise ValueError("The 'direction' parameter should be either 'min' or 'max'")

            if self.bounds:
                params[params < self.bounds[0]] = self.bounds[0]
                params[params > self.bounds[1]] = self.bounds[1]

            # update design
            self.optimization_problem.geometry.update_geometry(params=params)
            # self.optimization_problem.design_region.update_permittivity_vector()

            # save logs
            with open(self.log_file, 'a') as logs:
                logs.write('{}\t{}\t{}\n'.format(iteration, fom, transmission_coef))

            # save desgin
            self.save_results(current_iteration=iteration)
            self.plotter.plot_all(label=str(iteration) + '_', optimization=self.optimization_problem)

        # return the final parameters and gradient information
        return params, grad_adam, mopt, vopt

    def _step_adam(self, gradient, mopt_old, vopt_old, iteration, beta1, beta2, epsilon=1e-8):
        """ Performs one step of adam optimization"""

        mopt = beta1 * mopt_old + (1 - beta1) * gradient
        mopt_t = mopt / (1 - beta1 ** (iteration + 1))
        vopt = beta2 * vopt_old + (1 - beta2) * (np.square(gradient))
        vopt_t = vopt / (1 - beta2 ** (iteration + 1))
        grad_adam = mopt_t / (np.sqrt(vopt_t) + epsilon)

        return (grad_adam, mopt, vopt)

    def print_step(self, mode_objective_value, transmission_coef, time_used,
                   current_step, all_steps):
        print("Stage: {}/{}, Step: {}/{}, Time used: {:.2f} secs".format(self.current_stage, self.total_stage,
                                                                         current_step + 1,
                                                                         all_steps, time_used))

        print('current mode objective value: {}'.format(mode_objective_value))
        print('current transmission coefficients are: {}'.format(transmission_coef))

    def save_results(self, current_iteration):
        save_path = os.path.join(self.save_folder, 'saved_eps')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        with open(os.path.join(save_path, 'continuous_eps_{}.pkl'.format(current_iteration)), 'wb') as file:
            eps = self.optimization_problem.geometry.get_current_eps()
            # dump information to that file
            pickle.dump(eps, file)
        with open(os.path.join(save_path, 'continuous_params_{}.pkl'.format(current_iteration)), 'wb') as file:
            params = self.optimization_problem.geometry.get_current_params_inshape()
            beta = self.optimization_problem.geometry.beta
            filter_R = self.optimization_problem.geometry.filter_R
            # dump information to that file
            pickle.dump([params, beta, filter_R], file)

