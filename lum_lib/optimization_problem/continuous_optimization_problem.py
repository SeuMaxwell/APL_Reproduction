import os
import shutil
import inspect
import copy
import numpy as np
import matplotlib.pyplot as plt
import re
from autograd import grad, jacobian

from lum_lib.lumerical_simu_api.base_script import BaseScript
from lum_lib.utils.wavelengths import Wavelengths
from lum_lib.lumerical_simu_api.simulation import Simulation
from lum_lib.utils.gradients import GradientFields
from lum_lib.utils.plotter import Plotter
import lum_lib.lumerical_simu_api.scripts as scp


class Continuous_Optimization(object):
    """ New interface for adjoint optimization problems using continuous eps value.
    Flexible to define objective function and optimization process
    Calling the member function run will perform the optimization,
        which requires four key pieces:
            1) a script to generate the base simulation,                                  #基本脚本产生基本仿真
            2) an object that defines and collects the figure of merit,                   #优化函数定义
            3) an object that generates the shape under optimization for a given set of optimization parameters           #优化参数

        Parameters
        ----------
        :param base_script:    callable, file name or plain string with script to generate the base simulation.
        :param source_wavelengths:    wavelength value (float) or range (class Wavelengths) with the spectral range for all simulations.
        :param fom_list:            a list of figure of merit (class ModeMatch).
        :param fom_wavelength_id_list: a list of id indicating the wavelength of the fom calculation with respective to the source
        :param geometry:       optimizable geometry (class FunctionDefinedPolygon).
        :param objective_function:
        :param hide_fdtd_cad:  flag run FDTD CAD in the background.
        :param use_deps:       flag to use the numerical derivatives calculated directly from FDTD.
        :param plot_history:   plot the history of all parameters (and gradients)
        :param label:          If the optimization is part of a super-optimization, this string is used for the legend of the corresponding FOM plot
        :param source_name:    Name of the source object in the simulation project (default is "source")
    """

    def __init__(self, base_script, source_wavelengths, fom_list, fom_wavelength_id_list, geometry, objective_function, use_var_fdtd=False, hide_fdtd_cad=False,
                 use_deps=True, plot_history=True, label=None,  source_name='forward_source'):     #改了false
        self.base_script = base_script if isinstance(base_script, BaseScript) else BaseScript(base_script)
        self.source_wavelengths = source_wavelengths if isinstance(source_wavelengths, Wavelengths) else Wavelengths(source_wavelengths)
        self.fom_list = fom_list
        self.fom_wavelength_id_list = fom_wavelength_id_list

        self.geometry = geometry
        self.objective_function = objective_function

        self.use_var_fdtd = bool(use_var_fdtd)
        self.hide_fdtd_cad = bool(hide_fdtd_cad)
        self.source_name = source_name

        self.plot_history = plot_history

        self.plot_history = bool(plot_history)
        self.plotter = None  # < Initialize later, when we are done with adding FOMs
        self.old_dir = os.getcwd()
        self.full_fom_hist = []  # < Stores the result of every FOM evaluation
        self.params_hist = []  # < List of parameters after iterations
        self.grad_hist = []

        if callable(use_deps):
            self.use_deps = True
            self.custom_deps = use_deps
        else:
            self.use_deps = bool(use_deps)
            self.custom_deps = None

        self.unfold_symmetry = geometry.unfold_symmetry

        self.label = label

        if self.use_deps:
            print("Accurate interface detection enabled")

        # store the current fom and the corresponding coefficient
        self.current_fom = None
        self.fom_transmission_coeff_list = None

        ## Figure out from which file this method was called (most likely the driver script)
        frame = inspect.stack()[1]
        self.calling_file_name = os.path.abspath(frame[0].f_code.co_filename)
        # get the base path where the running python script is
        self.base_file_path = os.path.dirname(self.calling_file_name)

        # prepare current working directory
        working_dir = 'opts'
        self.prepare_working_dir(working_dir)

    def check_gradient_random(self, test_params, dx, num_gradients=1, working_dir=None):
        """
        pick a number of random design points, and compare the finite-difference gradient and adjoint gradient
        :param test_params:
        :param dx:
        :param num_gradients:
        :param working_dir:
        :return:
        """

        self.initialize()

        ## Calculate the gradient using the adjoint method:
        adj_grad = self.callable_jac(test_params)

        ## Calculate the gradient using finite differences
        fd_grad = list()
        # randomly choose indices to loop estimate
        fd_gradient_idx = np.random.choice(len(test_params), num_gradients, replace=False)

        cur_dx = dx / 2.
        for i in fd_gradient_idx:
            param = test_params[i]
            d_params = test_params.copy()
            d_params[i] = param + cur_dx
            f1 = self.callable_fom(d_params)
            # make sure f1 is a scalar for the later fitting process
            if not np.isscalar(f1):
                f1 = f1[0]

            d_params[i] = param - cur_dx
            f2 = self.callable_fom(d_params)
            # make sure f2 is a scalar for the later fitting process
            if not np.isscalar(f2):
                f2 = f2[0]

            fd_grad.append((f1 - f2) / dx)

        print('fd_gradient is: {}'.format(fd_grad))
        print('adj_gradient is: {}'.format(adj_grad[fd_gradient_idx]))

        (m, b) = np.polyfit(fd_grad, adj_grad[fd_gradient_idx], 1)

        min_g = np.min(fd_grad)
        max_g = np.max(fd_grad)

        plt.figure()
        plt.plot([min_g, max_g], [min_g, max_g], label='y=x comparison')
        plt.plot([min_g, max_g], [m * min_g + b, m * max_g + b], '--', label='Best fit')
        plt.plot(fd_grad, adj_grad[fd_gradient_idx], 'o', label='Adjoint comparison')
        plt.xlabel('Finite Difference Gradient')
        plt.ylabel('Adjoint Gradient')
        plt.title('comparison dx={}, N={}'.format(dx, num_gradients))
        plt.legend()
        plt.grid(True)
        plt.show()

        # # 添加更多分析
        # fd_grad_array = np.zeros_like(adj_grad)
        # fd_grad_array[fd_gradient_idx] = np.array(fd_grad)
        #
        # # 计算差异
        # diff = adj_grad - fd_grad_array
        # diff_norm = np.linalg.norm(diff[fd_gradient_idx])
        # print(f"梯度差异范数: {diff_norm}")
        #
        # # 绘制散点图和直方图
        # plt.figure(figsize=(12, 10))
        #
        # plt.subplot(2, 2, 1)
        # plt.plot([min_g, max_g], [min_g, max_g], label='y=x')
        # plt.plot([min_g, max_g], [m * min_g + b, m * max_g + b], '--', label='最佳拟合')
        # plt.plot(fd_grad, adj_grad[fd_gradient_idx], 'o', label='梯度比较')
        # plt.xlabel('有限差分梯度')
        # plt.ylabel('伴随梯度')
        # plt.title(f'梯度比较 dx={dx}, N={num_gradients}')
        # plt.legend()
        # plt.grid(True)
        #
        # plt.subplot(2, 2, 2)
        # plt.hist(diff[fd_gradient_idx], bins=20)
        # plt.xlabel('梯度差异')
        # plt.title('梯度差异分布')
        #
        # plt.subplot(2, 2, 3)
        # plt.hist(adj_grad[fd_gradient_idx], bins=20, alpha=0.5, label='伴随')
        # plt.hist(fd_grad, bins=20, alpha=0.5, label='有限差分')
        # plt.xlabel('梯度值')
        # plt.title('梯度分布比较')
        # plt.legend()
        #
        # # 保存图像
        # save_path = os.path.join(self.workingDir, 'gradient_check.png')
        # plt.savefig(save_path)
        # plt.close()
        #
        # return diff_norm

    def prepare_working_dir(self, working_dir):

        ## Check if we have an absolute path
        if not os.path.isabs(working_dir):
            ## If not, we assume it is meant relative to the path of the script which called this script
            working_dir = os.path.abspath(os.path.join(self.base_file_path, working_dir))

        ## Check if the provided path already ends with _xxxx (where xxxx is a number)
        result = re.match(r'_\d+$', working_dir)
        without_suffix = re.sub(r'_\d+$', '', working_dir)
        suffix_num = int(result[1:]) if result else 0
        working_dir = without_suffix + '_{}'.format(suffix_num)

        ## Check if path already exists. If so, keep increasing the number until it does not exist
        while os.path.exists(working_dir):
            suffix_num += 1
            working_dir = without_suffix + '_{}'.format(suffix_num)

        os.makedirs(working_dir)
        os.chdir(working_dir)

        ## Copy the calling script over
        if os.path.isfile(self.calling_file_name):
            shutil.copy(self.calling_file_name, working_dir)

        self.workingDir = working_dir

    def initialize(self):
        """
            Performs all steps that need to be carried only once at the beginning of the optimization.
        """

        ## Store a copy of the script file
        if hasattr(self.base_script, 'script_str'):
            with open('script_file.lsf', 'a') as file:
                file.write(self.base_script.script_str.replace(';', ';\n'))

        # create the FDTD CAD
        self.sim = Simulation(self.workingDir, self.use_var_fdtd, self.hide_fdtd_cad)
        self.geometry.check_license_requirements(self.sim)

        # FDTD model, set forward source wavelength as global source wavelength for forward simulation
        self.base_script(self.sim.fdtd)  # call base_script object to run the lumerical scripts
        self.set_global_wavelength(self.sim, self.source_wavelengths)

        # if multiple sources are defined, initialize all of them
        if isinstance(self.source_name, list):
            for source_name in self.source_name:
                self.set_source_wavelength(self.sim, source_name, self.fom_list[0].multi_freq_src,
                                           len(self.source_wavelengths))
        elif isinstance(self.source_name, str):
            self.set_source_wavelength(self.sim, self.source_name, self.fom_list[0].multi_freq_src, len(self.source_wavelengths))
        else:
            raise Exception('source name must be a string or a list of strings')

        # "opt_fields" is the name set in the .lsf file for the optimization region, fixed
        # add opt_fields_index to record the index
        self.sim.fdtd.setnamed('opt_fields', 'override global monitor settings', False)
        self.sim.fdtd.setnamed('opt_fields', 'spatial interpolation', 'none')
        self.add_index_monitor(self.sim, 'opt_fields')

        if self.use_deps:
            self.set_use_legacy_conformal_interface_detection(self.sim, False)

        start_params = self.geometry.get_current_params()

        # We need to add the geometry first because it adds the mesh override region
        self.geometry.add_geo(self.sim, start_params, only_update=False)

        for i, fom in enumerate(self.fom_list):
            fom.initialize(self.sim, self.source_wavelengths, self.fom_wavelength_id_list[i])

    def save_fields_to_vtk(self, cur_iteration):
        self.sim.save_fields_to_vtk(os.path.join(self.workingDir, 'global_fields_{}').format(cur_iteration))
        raise Exception('mw check here!')

    def save_index_to_vtk(self, cur_iteration):
        self.sim.save_index_to_vtk(os.path.join(self.workingDir, 'global_index_{}').format(cur_iteration))
        raise Exception('mw check here!')

    def make_forward_sim(self, params):

        self.sim.fdtd.switchtolayout()
        self.geometry.update_geometry(params)
        self.geometry.add_geo(self.sim, params=None, only_update=True)  # here the eps is updated in the simulation
        self.deactivate_all_sources(self.sim)

        # reset the forward simulation frequency data point number
        self.sim.fdtd.setglobalmonitor('frequency points', len(self.source_wavelengths))

        # if multiple sources are defined, activate all of them
        if isinstance(self.source_name, list):
            for source_name in self.source_name:
                self.sim.fdtd.setnamed(source_name, 'enabled', True)
        elif isinstance(self.source_name, str):
            self.sim.fdtd.setnamed(self.source_name, 'enabled', True)
        else:
            raise Exception('source name must be a string or a list of strings')


        for fom in self.fom_list:  # disable adjoint source
            fom.make_forward_sim(self.sim)

        forward_name = 'forward_simu'
        return self.sim.save(forward_name)

    def process_forward_sim(self):
        forward_name = 'forward_simu'
        self.sim.load(forward_name)
        self.check_simulation_was_successful(self.sim)

        # decouple the forward simulation results by wavelengths according to adjoint simulation
        self.forward_field_list = list()
        self.forward_field_wl_list = list()

        for i in range(len(self.fom_list)):
            forward_fields = scp.get_fields(self.sim.fdtd,
                                             monitor_name='opt_fields',
                                             field_result_name='forward_fields',
                                             get_eps=True,
                                             get_D=not self.use_deps,
                                             get_H=False,
                                             nointerpolation=not self.geometry.use_interpolation(),
                                             unfold_symmetry=self.unfold_symmetry,
                                            selected_wavelength_id=self.fom_wavelength_id_list[i])
            assert hasattr(forward_fields, 'E')
            forward_fields_wl = forward_fields.wl # get the wavelengths
            self.forward_field_list.append(forward_fields)
            self.forward_field_wl_list.append(forward_fields_wl)

        # self.forward_fields.E has the shape: (151, 151, 1, 11, 3) (Nx, Ny, Nz, wavelength_num, 3)
        self.fom_transmission_coeff_list = list()

        for fom in self.fom_list:
            self.fom_transmission_coeff_list.append(fom.get_fom_coefficient(self.sim))

        self.current_fom = self.objective_function(*self.fom_transmission_coeff_list)

        self.full_fom_hist.append(self.current_fom )

        print('FOM = {}'.format(self.current_fom ))

        return self.current_fom


    def make_adjoint_sim(self, params, index):
        assert np.allclose(params, self.geometry.get_current_params())
        adjoint_name = 'adjoint_simu_{}'.format(index)
        self.sim.fdtd.switchtolayout()
        self.geometry.add_geo(self.sim, params=None, only_update=True)

        # disable forward source
        # if multiple sources are defined, disable all of them
        if isinstance(self.source_name, list):
            for source_name in self.source_name:
                self.sim.fdtd.setnamed(source_name, 'enabled', False)
        elif isinstance(self.source_name, str):
            self.sim.fdtd.setnamed(self.source_name, 'enabled', False)
        else:
            raise Exception('source name must be a string or a list of strings')

        for i, fom in enumerate(self.fom_list):
            if i == index:  # activate adjoint source for the selected fom
                # make the default frequency points equal to the specific fom wavelength number
                self.sim.fdtd.setglobalmonitor('frequency points', len(self.fom_wavelength_id_list[i]))
                fom.make_adjoint_sim(self.sim)
            else:  # deactivate adjoint source for unselected foms
                fom.make_forward_sim(self.sim)

        return self.sim.save(adjoint_name)

    def process_adjoint_sim(self, index):
        adjoint_name = 'adjoint_simu_{}'.format(index)
        self.sim.load(adjoint_name)
        if self.sim.fdtd.layoutmode():
            self.sim.fdtd.run()
        self.check_simulation_was_successful(self.sim)

        adjoint_fields = scp.get_fields(self.sim.fdtd,
                                         monitor_name='opt_fields',
                                         field_result_name='adjoint_fields',
                                         get_eps=not self.use_deps,
                                         get_D=not self.use_deps,
                                         get_H=False,
                                         nointerpolation=not self.geometry.use_interpolation(),
                                         unfold_symmetry=self.unfold_symmetry)
        assert hasattr(adjoint_fields, 'E')

        dJ = jacobian(self.objective_function, index)(*self.fom_transmission_coeff_list)

        self.scaling_factor = self.fom_list[index].get_adjoint_field_scaling(self.sim, dJ=dJ)

        adjoint_fields.scale(3, self.scaling_factor)                         #插值

        self.adjoint_field_list.append(adjoint_fields)

    def callable_fom(self, params):
        """ Function for the optimizers to retrieve the figure of merit.
            :param params:  optimization parameters.
            :param returns: figure of merit.
        """

        self.sim.fdtd.clearjobs()
        print('Making forward solve')
        forward_job_name = self.make_forward_sim(params)
        self.sim.fdtd.addjob(forward_job_name)
        print('Running solves')
        self.sim.fdtd.runjobs()

        print('Processing forward solve')
        fom = self.process_forward_sim()

        return fom

    def callable_jac(self, params):
        """ Function for the optimizer to extract the figure of merit gradient.
            :param params:  optimization paramaters.
            :param returns: partial derivative of the figure of merit with respect to each optimization parameter.
        """

        self.sim.fdtd.clearjobs()

        # determine whether there is a forward field stored
        no_forward_fields = not hasattr(self, 'forward_fields')

        # determine whether the parameters stored is equal to the parameters obtained from the current geometry
        params_changed = not np.allclose(params, self.geometry.get_current_params())
        redo_forward_sim = no_forward_fields or params_changed

        self.adjoint_field_list = list()

        if redo_forward_sim:
            print('Making forward solver')
            # the forward simulation is saved as .fsp file
            forward_job_name = self.make_forward_sim(params)
            self.sim.fdtd.addjob(forward_job_name)

        print('Making adjoint solver')
        for adjoint_index in range(len(self.fom_list)):
            adjoint_job_name = self.make_adjoint_sim(params, index=adjoint_index)
            self.sim.fdtd.addjob(adjoint_job_name)

        if len(self.sim.fdtd.listjobs()) > 0:
            print('Running solvers')
            self.sim.fdtd.runjobs()

        # Take the E field out in the design region of forward simulation
        print('Processing forward solver')
        self.process_forward_sim()

        # Take the E field out in the design region of adjoint simulation
        print('Processing adjoint solver')
        for adjoint_index in range(len(self.fom_list)):
            self.process_adjoint_sim(index=adjoint_index)

        # caculate gradients from the E fields
        print('Calculating gradients')
        grad = self.calculate_adj_gradients()
        self.last_grad = grad

        return grad

    def calculate_adj_gradients(self):                                              #计算梯度
        """ Calculates the gradient of the figure of merit (FOM) with respect to each of the optimization parameters.
            It assumes that both the forward and adjoint solves have been run so that all the necessary field results
            have been collected.
        """
        self.gradient_field_list = list()
        for i in range(len(self.fom_list)):
            gradient_fields = GradientFields(forward_fields=self.forward_field_list[i], adjoint_fields=self.adjoint_field_list[i])
            self.gradient_field_list.append(gradient_fields)

        # directly calculate the gradient in python
        self.sim.fdtd.switchtolayout()

        gradient_list = list()

        for i in range(len(self.fom_list)):
            fom_partial_derivs_vs_wl = self.geometry.calculate_gradients(self.gradient_field_list[i])
            gradients = self.fom_list[i].fom_gradient_wavelength_integral(fom_partial_derivs_vs_wl,
                                                                   self.forward_field_wl_list[i])
            gradient_list.append(gradients)

        # when using np to sum up a list, equivalent to summing up a stacked np array, should include axis information!
        self.gradients = np.sum(gradient_list, axis=0)

        return self.gradients

    def plot_gradient_field(self, ax, title):
        """
        arbitrarily choose a gradient field object from list, and pass the overall gradient to the plot function
        :param ax:
        :param title:
        :return:
        """
        self.gradient_field_list[0].plot_gradients_from_opt(ax_gradients=ax, title=title, gradient_field=self.gradients)

    def make_final_sim(self, eps, save_path):
        """"""

        self.sim.fdtd.switchtolayout()

        # update final eps
        self.geometry.eps = eps

        self.geometry.add_geo(self.sim, params=None, only_update=False)  # here the eps is updated in the simulation

        self.deactivate_all_sources(self.sim)

        # reset the forward simulation frequency data point number
        self.sim.fdtd.setglobalmonitor('frequency points', len(self.source_wavelengths))
        self.sim.fdtd.setnamed(self.source_name, 'enabled', True)

        for fom in self.fom_list:  # disable adjoint source
            fom.make_forward_sim(self.sim)

        save_name = 'final_simu'
        return self.sim.save(os.path.join(save_path, save_name))

    @staticmethod
    def add_index_monitor(sim, monitor_name):
        """
        add the index monitor in the design region to record the index
        :param sim:
        :param monitor_name:
        :return:
        """
        sim.fdtd.select(monitor_name)
        if sim.fdtd.getnamednumber(monitor_name) != 1:
            raise UserWarning("a single object named '{}' must be defined in the base simulation.".format(monitor_name))
        index_monitor_name = monitor_name + '_index'
        if sim.fdtd.getnamednumber('FDTD') == 1:
            sim.fdtd.addindex()
        elif sim.fdtd.getnamednumber('varFDTD') == 1:
            sim.fdtd.addeffectiveindex()
        else:
            raise UserWarning('no FDTD or varFDTD solver object could be found.')
        sim.fdtd.set('name', index_monitor_name)
        sim.fdtd.setnamed(index_monitor_name, 'override global monitor settings', True)
        sim.fdtd.setnamed(index_monitor_name, 'frequency points', 1)
        sim.fdtd.setnamed(index_monitor_name, 'record conformal mesh when possible', True)
        monitor_type = sim.fdtd.getnamed(monitor_name, 'monitor type')
        geometric_props = ['monitor type']
        geometric_props.extend(Continuous_Optimization.cross_section_monitor_props(monitor_type))
        for prop_name in geometric_props:
            prop_val = sim.fdtd.getnamed(monitor_name, prop_name)
            sim.fdtd.setnamed(index_monitor_name, prop_name, prop_val)
        sim.fdtd.setnamed(index_monitor_name, 'spatial interpolation', 'none')

    @staticmethod
    def cross_section_monitor_props(monitor_type):
        geometric_props = ['x', 'y', 'z']
        if monitor_type == '3D':
            geometric_props.extend(['x span', 'y span', 'z span'])
        elif monitor_type == '2D X-normal':
            geometric_props.extend(['y span', 'z span'])
        elif monitor_type == '2D Y-normal':
            geometric_props.extend(['x span', 'z span'])
        elif monitor_type == '2D Z-normal':
            geometric_props.extend(['x span', 'y span'])
        elif monitor_type == 'Linear X':
            geometric_props.append('x span')
        elif monitor_type == 'Linear Y':
            geometric_props.append('y span')
        elif monitor_type == 'Linear Z':
            geometric_props.append('z span')
        else:
            raise UserWarning('monitor should be 2D or linear for a mode expansion to be meaningful.')
        return geometric_props

    @staticmethod
    def set_global_wavelength(sim, wavelengths):
        sim.fdtd.setglobalmonitor('use source limits', True)
        sim.fdtd.setglobalmonitor('use linear wavelength spacing', True)
        sim.fdtd.setglobalmonitor('frequency points', len(wavelengths))
        sim.fdtd.setglobalsource('set wavelength', True)
        sim.fdtd.setglobalsource('wavelength start', wavelengths.min())
        sim.fdtd.setglobalsource('wavelength stop', wavelengths.max())

    @staticmethod
    def set_source_wavelength(sim, source_name, multi_freq_src, freq_pts):
        if sim.fdtd.getnamednumber(source_name) != 1:
            raise UserWarning("a single object named '{}' must be defined in the base simulation.".format(source_name))
        if sim.fdtd.getnamed(source_name, 'override global source settings'):
            print('Wavelength range of source object will be superseded by the global settings.')
        sim.fdtd.setnamed(source_name, 'override global source settings', False)
        sim.fdtd.select(source_name)
        if sim.fdtd.haveproperty('multifrequency mode calculation'):
            sim.fdtd.setnamed(source_name, 'multifrequency mode calculation', multi_freq_src)
            if multi_freq_src:
                sim.fdtd.setnamed(source_name, 'frequency points', freq_pts)
        elif sim.fdtd.haveproperty('multifrequency beam calculation'):
            sim.fdtd.setnamed(source_name, 'multifrequency beam calculation', multi_freq_src)
            if multi_freq_src:
                sim.fdtd.setnamed(source_name, 'number of frequency points', freq_pts)

    @staticmethod
    def set_use_legacy_conformal_interface_detection(sim, flagVal):
        if sim.fdtd.getnamednumber('FDTD') == 1:
            sim.fdtd.select('FDTD')
        elif sim.fdtd.getnamednumber('varFDTD') == 1:
            sim.fdtd.select('varFDTD')
        else:
            raise UserWarning('no FDTD or varFDTD solver object could be found.')
        if bool(sim.fdtd.haveproperty('use legacy conformal interface detection')):
            sim.fdtd.set('use legacy conformal interface detection', flagVal)
            sim.fdtd.set('conformal meshing refinement', 51)
            sim.fdtd.set('meshing tolerance', 1.0 / 1.134e14)
        else:
            raise UserWarning(
                'install a more recent version of FDTD or the permittivity derivatives will not be accurate.')

    @staticmethod
    def check_simulation_was_successful(sim):
        if sim.fdtd.getnamednumber('FDTD') == 1:
            simulation_status = sim.fdtd.getresult('FDTD', 'status')
        elif sim.fdtd.getnamednumber('varFDTD') == 1:
            simulation_status = sim.fdtd.getresult('varFDTD', 'status')
        else:
            raise UserWarning('no FDTD or varFDTD solver object could be found.')
        if simulation_status != 1 and simulation_status != 2:  # run to full simulation time (1) or autoshutoff triggered (2)
            raise UserWarning('FDTD simulation did not complete successfully: status {0}'.format(simulation_status))
        return simulation_status

    @staticmethod
    def deactivate_all_sources(sim):
        sim.fdtd.selectall()
        numElements = int(sim.fdtd.getnumber())
        for i in range(numElements):
            objType = sim.fdtd.get("type", i + 1)
            if "Source" in objType:
                sim.fdtd.set("enabled", False, i + 1)