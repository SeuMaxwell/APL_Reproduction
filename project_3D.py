######## IMPORTS ########
# General purpose imports
import numpy as np
from autograd import numpy as npa
import os
import sys
import scipy as sp
import pickle

# Optimization specific imports
from lum_lib.design_region.topology import TopologyOptimization3DLayered
from lum_lib.lumerical_simu_api.load_lumerical_scripts import load_from_lsf
from lum_lib.figure_of_merit.modematch import ModeMatch
from lum_lib.optimization_problem.continuous_optimization_problem import Continuous_Optimization
from lum_lib.optimizers.continuous_optimizers import Adam_Optimizer
from lum_lib.utils.wavelengths import Wavelengths


######## RUNS TOPOLOGY OPTIMIZATION OF A 2D STRUCTURE ########
def runSim_Adam(params, eps_min, eps_max, x_pos, y_pos,z_pos, filter_R, objective_function):
    ######## DEFINE A 2D TOPOLOGY OPTIMIZATION REGION ########
    geometry = TopologyOptimization3DLayered(params=params, eps_min=eps_min, eps_max=eps_max, x=x_pos, y=y_pos, z=z_pos,
                                      eta=0.5, beta=1, beta_factor=2, filter_R=filter_R)

    ######## DEFINE FIGURE OF MERIT ########
    source_wavelengths = Wavelengths(start=1475e-9, stop=1625e-9, points=15)

    upper_port_wavelength_ids = np.arange(1, 2)  # 上端口FOM波长点
    lower_port_wavelength_ids = np.arange(13, 14)  # 下端口FOM波长点

    # 定义上下端口的模式匹配目标
    mode_fom_dict = {'names': ['fom1', 'fom2'],
                     'mode_numbers': ['Fundamental TE mode', 'Fundamental TE mode'],
                     'direction': ['Forward', 'Forward'],
                     'norm_p': [2, 2]}

    # 上端口：目标是最大化传输
    upper_port_fom = ModeMatch(monitor_name=mode_fom_dict['names'][0],
                               mode_number=mode_fom_dict['mode_numbers'][0],
                               direction=mode_fom_dict['direction'][0],
                               norm_p=mode_fom_dict['norm_p'][0])

    # 下端口：目标是最小化传输（在目标函数中实现）
    lower_port_fom = ModeMatch(monitor_name=mode_fom_dict['names'][1],
                               mode_number=mode_fom_dict['mode_numbers'][1],
                               direction=mode_fom_dict['direction'][1],
                               norm_p=mode_fom_dict['norm_p'][1])

    ######## LOAD TEMPLATE SCRIPT AND SUBSTITUTE PARAMETERS ########
    cur_path = os.path.dirname(os.path.abspath(__file__))
    script = load_from_lsf(os.path.join(cur_path, 'APL.lsf'))  # 需要相应地修改仿真脚本

    fom_list = [upper_port_fom, lower_port_fom]
    fom_wavelength_id_list = [upper_port_wavelength_ids, lower_port_wavelength_ids]

    opt = Continuous_Optimization(base_script=script, source_wavelengths=source_wavelengths,
                                  fom_list=fom_list, fom_wavelength_id_list=fom_wavelength_id_list,
                                  geometry=geometry, objective_function=objective_function,
                                  use_deps=False, hide_fdtd_cad=False, plot_history=False)

    ######## RUN THE OPTIMIZER ########
    # 多阶段优化，逐步提高beta值以增强二值化
    step_iter_list = [30, 30, 30, 30, 30, 30, 30, 30, 30]
    step_num_list = list()
    for i in range(len(step_iter_list)):
        step_num_list.append(np.sum(step_iter_list[:i + 1]))

    optimizer_adam = Adam_Optimizer(optimization_problem=opt,
                                    step_num_list=step_num_list,
                                    step_size=1e-2,
                                    bounds=(0, 1),
                                    direction='max',
                                    device_name='APL')
    optimizer_adam.run_optimization_stage()


def check_grad(params, eps_min, eps_max, x_pos, y_pos, z_pos,filter_R, objective_function):
    ######## DEFINE A 2D TOPOLOGY OPTIMIZATION REGION ########
    geometry = TopologyOptimization3DLayered(params=params, eps_min=eps_min, eps_max=eps_max, x=x_pos, y=y_pos, z=z_pos,
                                             eta=0.5, beta=1, beta_factor=2, filter_R=filter_R)

    ######## DEFINE FIGURE OF MERIT ########
    source_wavelengths = Wavelengths(start=1450e-9, stop=1650e-9, points=40)

    upper_port_wavelength_ids = np.arange(0, 10)  # 上端口FOM波长点
    lower_port_wavelength_ids = np.arange(30, 40)  # 下端口FOM波长点

    # 定义上下端口的模式匹配目标
    mode_fom_dict = {'names': ['fom1', 'fom2'],
                     'mode_numbers': ['Fundamental TE mode', 'Fundamental TE mode'],
                     'direction': ['Forward', 'Forward'],
                     'norm_p': [2, 2]}

    # 上端口：目标是最大化传输
    upper_port_fom = ModeMatch(monitor_name=mode_fom_dict['names'][0],
                               mode_number=mode_fom_dict['mode_numbers'][0],
                               direction=mode_fom_dict['direction'][0],
                               norm_p=mode_fom_dict['norm_p'][0])

    # 下端口：目标是最小化传输（在目标函数中实现）
    lower_port_fom = ModeMatch(monitor_name=mode_fom_dict['names'][1],
                               mode_number=mode_fom_dict['mode_numbers'][1],
                               direction=mode_fom_dict['direction'][1],
                               norm_p=mode_fom_dict['norm_p'][1])

    ######## LOAD TEMPLATE SCRIPT AND SUBSTITUTE PARAMETERS ########
    cur_path = os.path.dirname(os.path.abspath(__file__))
    script = load_from_lsf(os.path.join(cur_path, 'APL.lsf'))

    fom_list = [upper_port_fom, lower_port_fom]
    fom_wavelength_id_list = [upper_port_wavelength_ids, lower_port_wavelength_ids]

    opt = Continuous_Optimization(base_script=script, source_wavelengths=source_wavelengths,
                                  fom_list=fom_list, fom_wavelength_id_list=fom_wavelength_id_list,
                                  geometry=geometry, objective_function=objective_function,
                                  use_deps=False, hide_fdtd_cad=False, plot_history=False)

    ######## Check gradient ########
    test_para = params
    opt.check_gradient_random(test_params=test_para, dx=1e-2, num_gradients=5, working_dir=None)

def get_simu_results(eps_path, eps_min, eps_max, x_pos, y_pos,z_pos, filter_R, objective_function):
    ######## DEFINE A 2D TOPOLOGY OPTIMIZATION REGION ########
    ######## DEFINE A 2D TOPOLOGY OPTIMIZATION REGION ########
    geometry = TopologyOptimization3DLayered(params=None, eps_min=eps_min, eps_max=eps_max, x=x_pos, y=y_pos, z=z_pos,
                                             eta=0.5, beta=1, beta_factor=2, filter_R=filter_R)

    ######## DEFINE FIGURE OF MERIT ########
    # 设置宽带光源波长范围：1530nm-1560nm，采用多点采样实现宽带优化
    # source_wavelengths = Wavelengths(start=1310e-9, stop=1550e-9, points=2)
    source_wavelengths = Wavelengths(start=1260e-9, stop=1600e-9, points=35)

    # fom_wavelength_id should be an array to be consistent with the dimension routine
    # upper_port_wavelength_ids = np.array([0])
    # lower_port_wavelength_ids = np.array([1])
    # 所有波长点都参与优化计算
    upper_port_wavelength_ids = np.arange(0, 11)  # 上端口FOM波长点
    lower_port_wavelength_ids = np.arange(0, 11)  # 下端口FOM波长点

    # 定义上下端口的模式匹配目标
    mode_fom_dict = {'names': ['fom1', 'fom2'],
                     'mode_numbers': ['Fundamental TE mode', 'Fundamental TE mode'],
                     'direction': ['Forward', 'Forward'],
                     'norm_p': [2, 2]}

    # 上端口：目标是最大化传输
    upper_port_fom = ModeMatch(monitor_name=mode_fom_dict['names'][0],
                               mode_number=mode_fom_dict['mode_numbers'][0],
                               direction=mode_fom_dict['direction'][0],
                               norm_p=mode_fom_dict['norm_p'][0])

    # 下端口：目标是最小化传输（在目标函数中实现）
    lower_port_fom = ModeMatch(monitor_name=mode_fom_dict['names'][1],
                               mode_number=mode_fom_dict['mode_numbers'][1],
                               direction=mode_fom_dict['direction'][1],
                               norm_p=mode_fom_dict['norm_p'][1])

    ######## LOAD TEMPLATE SCRIPT AND SUBSTITUTE PARAMETERS ########
    cur_path = os.path.dirname(os.path.abspath(__file__))
    script = load_from_lsf(os.path.join(cur_path, 'APL.lsf'))  # 需要相应地修改仿真脚本

    fom_list = [upper_port_fom, lower_port_fom]
    fom_wavelength_id_list = [upper_port_wavelength_ids, lower_port_wavelength_ids]

    opt = Continuous_Optimization(base_script=script, source_wavelengths=source_wavelengths,
                                  fom_list=fom_list, fom_wavelength_id_list=fom_wavelength_id_list,
                                  geometry=geometry, objective_function=objective_function,
                                  use_deps=False, hide_fdtd_cad=False, plot_history=False)
    opt.initialize()
    with open(os.path.join(eps_path, 'continuous_eps_172.pkl'), 'rb') as file:
        eps_final = pickle.load(file)

    # binarize
    eps_final[eps_final > (eps_min+eps_max)/2] = eps_max
    eps_final[eps_final <= (eps_min+eps_max)/2] = eps_min

    opt.make_final_sim(eps_final, save_path='../')


if __name__ == '__main__':
    def J(upper, lower):
        # 基本参数
        alpha = 1.0  # 上端口透过率权重
        beta = 1.0  # 下端口抑制权重

        # 计算各端口平均透过率
        upper_mean = npa.mean(upper)
        lower_mean = npa.mean(lower)

        # 上端口传输项：优化上端口透过率接近1
        upper_transmission_term = alpha * (1.0 - (1.0 - upper_mean) ** 2)

        # 下端口传输项：同样鼓励下端口透过率接近1
        lower_transmission_term = beta * (1.0 - (1.0 - lower_mean) ** 2)
        # 计算透过率目标函数

        # 总目标函数 - 两个端口都接近1的透过率
        return upper_transmission_term + lower_transmission_term


    # 设计区域参数设置
    size_x = 40000  # 设计区长度 (nm)
    delta_x = 100  # x轴像素大小 (nm)

    size_y = 8500  # 设计区宽度 (nm)
    delta_y = 100  # y轴像素大小 (nm)

    deepth = 330

    size_z = 30     #pcm厚度
    delta_z = 30

    filter_R = 500  # 平滑滤波半径 (nm)，用于移除小特征，提高可制造性

    # 材料参数
    eps_max = 4.050 ** 2
    eps_min = 3.285 ** 2

    # 计算设计区网格点数
    x_points = int(size_x / delta_x) + 1
    y_points = int(size_y / delta_y) + 1
    z_points = int(size_z / delta_z) + 1

    # 创建坐标网格
    x_pos = np.linspace(-size_x / 2, size_x / 2, x_points) * 1e-9
    y_pos = np.linspace(-size_y / 2, size_y / 2, y_points) * 1e-9
    z_pos = np.linspace(deepth / 2, deepth / 2+ size_z, z_points)* 1e-9

    # 设置随机数种子以便结果可复现
    seed = 666
    np.random.seed(seed)

    # 创建初始随机参数
    initial_cond = np.random.rand(x_points * y_points)

    # 取消注释以检查梯度计算
    check_grad(initial_cond, eps_min, eps_max, x_pos, y_pos, z_pos,filter_R * 1e-9, objective_function=J)

    # 运行优化
    # runSim_Adam(initial_cond, eps_min, eps_max, x_pos, y_pos, z_pos,filter_R * 1e-9, objective_function=J)
    # eps_path = 'C:\\Users\\ZWR\\Desktop\\Reconfigurable\\WDM\\opts_40\\results\\saved_eps'
    # get_simu_results(eps_path, eps_min, eps_max, x_pos, y_pos,z_pos, filter_R * 1e-9, objective_function=J)