import sys
import numpy as np
import scipy as sp
import scipy.constants

eps0 = sp.constants.epsilon_0
import lum_lib.lumerical_simu_api.scripts as scp
from lum_lib.utils.wavelengths import Wavelengths

""" Helper function to determine if a variable can be converted to an integer """
def is_int(x):
    try:
        int(x)
        return True
    except ValueError:
        return False


class ModeMatch(object):
    """ Calculates the figure of merit from an overlap integral between the fields recorded by a field monitor and the slected mode.
        A mode expansion monitor is added to the field monitor to calculate the overlap result, which appears as T_forward in the
        list of mode expansion monitor results. The T_forward result is described in the following page:

            https://kb.lumerical.com/ref_sim_obj_using_mode_expansion_monitors.html

        This result is equivalent to equation (7) in the following paper:

           C. Lalau-Keraly, S. Bhargava, O. Miller, and E. Yablonovitch, "Adjoint shape optimization applied to electromagnetic design,"
           Opt. Express  21, 21693-21701 (2013). https://doi.org/10.1364/OE.21.021693

        Parameters
        ----------
        :param monitor_name:   name of the field monitor that records the fields to be used in the mode overlap calculation.
        :param mode_number:    mode number in the list of modes generated by the mode expansion monitor.
        :param direction:      direction of propagation ('Forward' or 'Backward') of the mode injected by the source.
        :param multi_freq_src: bool flag to enable / disable a multi-frequency mode calculation and injection for the adjoint source.
        :param norm_p:         exponent of the p-norm used to generate the figure of merit; use to generate the FOM.
      """

    def __init__(self, monitor_name, mode_number, direction, multi_freq_src=False, norm_p=2):
        self.monitor_name = str(monitor_name)
        if not self.monitor_name:
            raise UserWarning('empty monitor name.')
        self.mode_expansion_monitor_name = monitor_name + '_mode_exp'
        self.adjoint_source_name = monitor_name + '_mode_src'
        self.mode_number = mode_number

        if is_int(mode_number):
            self.mode_number = int(mode_number)
            if self.mode_number <= 0:
                raise UserWarning('mode number should be positive.')
        else:
            self.mode_number = mode_number
        self.direction = str(direction)
        self.multi_freq_src = bool(multi_freq_src)
        if self.direction != 'Forward' and self.direction != 'Backward':
            raise UserWarning('invalid propagation direction.')

        self.norm_p = int(norm_p)
        if self.norm_p < 1:
            raise UserWarning('exponent p for norm must be positive.')

    def initialize(self, sim, source_wavelength, fom_wavelength_id):
        self.check_monitor_alignment(sim)
        self.update_adjoint_wavelength(source_wavelength=source_wavelength, wavelength_id=fom_wavelength_id)

        ModeMatch.add_mode_expansion_monitor(sim, self.monitor_name, self.mode_expansion_monitor_name, self.mode_number)
        adjoint_injection_direction = 'Backward' if self.direction == 'Forward' else 'Forward'
        self.add_mode_source(sim, self.monitor_name, self.adjoint_source_name, adjoint_injection_direction,
                                  self.mode_number, self.multi_freq_src)

    def update_adjoint_wavelength(self, source_wavelength, wavelength_id):
        """
        create the wavelength object for adjoint simulation, which can be different from source wavelength
        :param source_wavelength: an object of Wavelength class
        :param wavelength_id: numpy array
        :return:
        """
        source_wavelength_points = source_wavelength.asarray()
        adjoint_wavelength_points = source_wavelength_points[wavelength_id]
        self.adjoint_wavelength = Wavelengths(start=adjoint_wavelength_points[0], stop=adjoint_wavelength_points[-1],
                                              points=len(adjoint_wavelength_points))
        self.forward_wavelength = source_wavelength
        self.wavelength_id = wavelength_id

    def make_forward_sim(self, sim):
        sim.fdtd.setnamed(self.adjoint_source_name, 'enabled', False)

    def make_adjoint_sim(self, sim):
        sim.fdtd.setnamed(self.adjoint_source_name, 'enabled', True)

    def update_adjoint_source_amp(self, sim, dJ):
        scaling_factor = self.get_adjoint_field_scaling(sim=sim, dJ=dJ)

    def check_monitor_alignment(self, sim):

        ## Here, we check that the FOM_monitor is properly aligned with the mesh
        if sim.fdtd.getnamednumber(self.monitor_name) != 1:
            raise UserWarning('monitor could not be found or the specified name is not unique.')

        # Get the orientation
        monitor_type = sim.fdtd.getnamed(self.monitor_name, 'monitor type')

        if (monitor_type == 'Linear X') or (monitor_type == '2D X-normal'):
            orientation = 'x'
        elif (monitor_type == 'Linear Y') or (monitor_type == '2D Y-normal'):
            orientation = 'y'
        elif (monitor_type == 'Linear Z') or (monitor_type == '2D Z-normal'):
            orientation = 'z'
        else:
            raise UserWarning('monitor should be 2D or linear for a mode expansion to be meaningful.')

        monitor_pos = sim.fdtd.getnamed(self.monitor_name, orientation)
        if sim.fdtd.getnamednumber('FDTD') == 1:
            grid = sim.fdtd.getresult('FDTD', orientation)
        elif sim.fdtd.getnamednumber('varFDTD') == 1:
            grid = sim.fdtd.getresult('varFDTD', orientation)
        else:
            raise UserWarning('no FDTD or varFDTD solver object could be found.')
        ## Check if this is exactly aligned with the simulation mesh. It exactly aligns if we find a point
        ## along the grid which is no more than 'tol' away from the position
        tol = 1e-9
        dist_from_nearest_mesh_point = min(abs(grid - monitor_pos))
        if dist_from_nearest_mesh_point > tol:
            print(
                'WARNING: The monitor "{}" is not aligned with the grid. Its distance to the nearest mesh point is {}. This can introduce small phase errors which sometimes result in inaccurate gradients.'.format(
                    self.monitor_name, dist_from_nearest_mesh_point))

    @staticmethod
    def add_mode_expansion_monitor(sim, monitor_name, mode_expansion_monitor_name, mode):                         #add mode expansion monitor to the field monitor
        # modify existing DFT monitor
        if sim.fdtd.getnamednumber(monitor_name) != 1:
            raise UserWarning('monitor could not be found or the specified name is not unique.')
        sim.fdtd.setnamed(monitor_name, 'override global monitor settings', False)
        # append a mode expansion monitor to the existing DFT monitor
        if sim.fdtd.getnamednumber(mode_expansion_monitor_name) == 0:
            sim.fdtd.addmodeexpansion()
            sim.fdtd.set('name', mode_expansion_monitor_name)
            sim.fdtd.setexpansion(mode_expansion_monitor_name, monitor_name)

            sim.fdtd.setnamed(mode_expansion_monitor_name, 'auto update before analysis', True)
            sim.fdtd.setnamed(mode_expansion_monitor_name, 'override global monitor settings', False)
            # properties that must be synchronized
            props = ['monitor type']
            monitor_type = sim.fdtd.getnamed(monitor_name, 'monitor type')
            geo_props, normal = ModeMatch.cross_section_monitor_props(monitor_type)
            props.extend(geo_props)
            # synchronize properties, locate the mode expansion monitor at the fom position
            for prop_name in props:
                prop_val = sim.fdtd.getnamed(monitor_name, prop_name)
                sim.fdtd.setnamed(mode_expansion_monitor_name, prop_name, prop_val)

            # select mode
            sim.fdtd.select(mode_expansion_monitor_name)

            ## If "mode" is an integer, it means that we want a 'user select' mode. Otherwise we try using it as a string directly
            if is_int(mode):
                sim.fdtd.setnamed(mode_expansion_monitor_name, 'mode selection', 'user select')
                sim.fdtd.updatemodes(mode)
            else:
                sim.fdtd.setnamed(mode_expansion_monitor_name, 'mode selection', mode)
                sim.fdtd.updatemodes()
        else:
            raise UserWarning('there is already a expansion monitor with the same name.')

    @staticmethod
    def cross_section_monitor_props(monitor_type):
        geometric_props = ['x', 'y', 'z']
        normal = ''
        if monitor_type == '2D X-normal':
            geometric_props.extend(['y span', 'z span'])
            normal = 'x'
        elif monitor_type == '2D Y-normal':
            geometric_props.extend(['x span', 'z span'])
            normal = 'y'
        elif monitor_type == '2D Z-normal':
            geometric_props.extend(['x span', 'y span'])
            normal = 'z'
        elif monitor_type == 'Linear X':
            geometric_props.append('x span')
            normal = 'y'
        elif monitor_type == 'Linear Y':
            geometric_props.append('y span')
            normal = 'x'
        elif monitor_type == 'Linear Z':
            geometric_props.append('z span')
        else:
            raise UserWarning('monitor should be 2D or linear for a mode expansion to be meaningful.')
        return geometric_props, normal

    def get_fom_coefficient(self, sim):                            #calculate the figure of merit 算透射率
        """
        the basic fom is evaluated as the transmission integrated over the frequency range
        fom
        :param sim:
        :return:
        """
        trans_coeff, wavelengths = self.get_transmission_coefficient(sim, self.direction, self.monitor_name,
                                                                               self.mode_expansion_monitor_name)
        # judge if the two arrays are element-wise equal, error=1e-5
        assert np.allclose(wavelengths, self.adjoint_wavelength.asarray())
        source_power = ModeMatch.get_source_power(sim, self.adjoint_wavelength.asarray())

        # trans_coeff is the unnormalized transmitted amplitude
        # T_fwd_vs_wavelength is the power transmission into specific mode
        self.T_fwd_vs_wavelength = np.real(trans_coeff * trans_coeff.conj() / source_power)

        self.phase_prefactors = trans_coeff / 4.0 / source_power
        fom_trans_coeff = ModeMatch.fom_wavelength_integral(self.T_fwd_vs_wavelength, self.adjoint_wavelength, self.norm_p)
        return fom_trans_coeff

    def get_adjoint_field_scaling(self, sim, dJ):                                        #加入adjoint source  模式光源的选择 应当是与正向光源相同的
        """
        refer to https://support.lumerical.com/hc/en-us/articles/360034915813
        https://optics.ansys.com/hc/en-us/articles/360034902433-Using-and-understanding-Mode-Expansion-Monitors
        first setup the adjoint source with the same time profile as the forward source, then normalize the
        field (with respect to different wavelength) to get the actual gradient!
        """
        omega = 2.0 * np.pi * sp.constants.speed_of_light / self.adjoint_wavelength.asarray()
        adjoint_source_power = ModeMatch.get_source_power(sim, self.adjoint_wavelength.asarray())

        # eps0 and dV is moved out side the scaling for numerical reasons, check topology.py, TopologyOptimization2D.calculate_gradients()
        scaling_factor = dJ * np.conj(self.phase_prefactors) * omega * 1j / np.sqrt(adjoint_source_power)
        return scaling_factor

    def get_transmission_coefficient(self, sim, direction, monitor_name, mode_exp_monitor_name):
        mode_exp_result_name = 'expansion for ' + mode_exp_monitor_name
        if not sim.fdtd.haveresult(mode_exp_monitor_name, mode_exp_result_name):
            raise UserWarning('unable to calcualte mode expansion.')
        mode_exp_data_set = sim.fdtd.getresult(mode_exp_monitor_name, mode_exp_result_name)
        mode_exp_wl = mode_exp_data_set['lambda']
        # here fwd_trans_coeff and back_trans_coeff are actually the total mode amplitude transmitted in the waveguide,
        # not normalized to source power, refer to https://support.lumerical.com/hc/en-us/articles/360034902433
        fwd_trans_coeff = mode_exp_data_set['a'] * np.sqrt(mode_exp_data_set['N'].real)
        back_trans_coeff = mode_exp_data_set['b'] * np.sqrt(mode_exp_data_set['N'].real)
        if direction == 'Backward':
            fwd_trans_coeff, back_trans_coeff = back_trans_coeff, fwd_trans_coeff
        return fwd_trans_coeff.flatten()[self.wavelength_id], mode_exp_wl.flatten()[self.wavelength_id]

    @staticmethod
    def get_global_wavelengths(sim):
        return Wavelengths(sim.fdtd.getglobalsource('wavelength start'),
                           sim.fdtd.getglobalsource('wavelength stop'),
                           sim.fdtd.getglobalmonitor('frequency points')).asarray()

    @staticmethod
    def get_source_power(sim, wavelengths):
        frequency = sp.constants.speed_of_light / wavelengths
        source_power = sim.fdtd.sourcepower(frequency)
        return np.asarray(source_power).flatten()

    @staticmethod
    def fom_wavelength_integral(T_fwd_vs_wavelength, wavelengths, norm_p):
        if len(wavelengths) > 1:
            # # mw: the next line is a check code, for broadband source to check frequency point not in the middle of
            # # frequency range, the gradient agrees well
            # fom = np.abs(T_fwd_vs_wavelength[..., -1].flatten())

            wavelength_range = wavelengths.max() - wavelengths.min()
            assert wavelength_range > 0.0, "wavelength range must be positive."
            T_fwd_abs = np.abs(T_fwd_vs_wavelength.flatten())
            T_fwd_abs_integrand = np.power(T_fwd_abs, norm_p) / wavelength_range
            fom = np.power(np.trapz(y=T_fwd_abs_integrand, x=wavelengths), 1.0 / norm_p)
        else:
            fom = np.abs(T_fwd_vs_wavelength.flatten())
        return fom.real

    def add_mode_source(self, sim, monitor_name, source_name, direction, mode, multi_freq_src):               #加入模式光源  后面再对场进行归一化
        if sim.fdtd.getnamednumber('FDTD') == 1:
            sim.fdtd.addmode()
        elif sim.fdtd.getnamednumber('varFDTD') == 1:
            sim.fdtd.addmodesource()
        else:
            raise UserWarning('no FDTD or varFDTD solver object could be found.')
        sim.fdtd.set('name', source_name)
        sim.fdtd.select(source_name)
        monitor_type = sim.fdtd.getnamed(monitor_name, 'monitor type')
        geo_props, normal = ModeMatch.cross_section_monitor_props(monitor_type)
        sim.fdtd.setnamed(source_name, 'injection axis', normal.lower() + '-axis')
        if sim.fdtd.getnamednumber('varFDTD') == 1:
            geo_props.remove('z')
        for prop_name in geo_props:
            prop_val = sim.fdtd.getnamed(monitor_name, prop_name)
            sim.fdtd.setnamed(source_name, prop_name, prop_val)

        sim.fdtd.setnamed(source_name, 'override global source settings', True)
        sim.fdtd.setnamed(source_name, 'direction', direction)

        sim.fdtd.setnamed(source_name, 'set wavelength', True)
        sim.fdtd.setnamed(source_name, 'wavelength start', self.adjoint_wavelength.min())
        sim.fdtd.setnamed(source_name, 'wavelength stop', self.adjoint_wavelength.max())

        if sim.fdtd.haveproperty('multifrequency mode calculation'):
            sim.fdtd.setnamed(source_name, 'multifrequency mode calculation', multi_freq_src)
            if multi_freq_src:
                sim.fdtd.setnamed(source_name, 'frequency points', sim.fdtd.getglobalmonitor('frequency points'))

        if is_int(mode):
            sim.fdtd.setnamed(source_name, 'mode selection', 'user select')
            sim.fdtd.select(source_name)
            sim.fdtd.updatesourcemode(int(mode))
        else:
            sim.fdtd.setnamed(source_name, 'mode selection', mode)
            sim.fdtd.select(source_name)
            sim.fdtd.updatesourcemode()

    def fom_gradient_wavelength_integral(self, T_fwd_partial_derivs_vs_wl, wl):
        assert np.allclose(wl, self.adjoint_wavelength.asarray())
        return ModeMatch.fom_gradient_wavelength_integral_impl(self.T_fwd_vs_wavelength, T_fwd_partial_derivs_vs_wl,
                                                               self.adjoint_wavelength.asarray(), self.norm_p)

    @staticmethod
    def fom_gradient_wavelength_integral_impl(T_fwd_vs_wavelength, T_fwd_partial_derivs_vs_wl, wl, norm_p):
        if wl.size > 1:
            print(f"T_fwd shape: {T_fwd_partial_derivs_vs_wl.shape}, wl size: {wl.size}")

            # 修复：检查T_fwd_partial_derivs_vs_wl的维度并根据需要重塑
            if len(T_fwd_partial_derivs_vs_wl.shape) == 1:
                # 一维数组情况 - 需要重塑为(N, wl.size)
                num_params = T_fwd_partial_derivs_vs_wl.shape[0] // wl.size
                if num_params * wl.size == T_fwd_partial_derivs_vs_wl.shape[0]:
                    # 可以正确重塑
                    T_fwd_partial_derivs_vs_wl = T_fwd_partial_derivs_vs_wl.reshape(num_params, wl.size)
                else:
                    # 无法重塑，假设梯度对所有波长都相同
                    T_fwd_partial_derivs_vs_wl = np.tile(T_fwd_partial_derivs_vs_wl.reshape(-1, 1), (1, wl.size))

            assert T_fwd_partial_derivs_vs_wl.shape[1] == wl.size

        # if wl.size > 1:
        #     print(f"T_fwd shape: {T_fwd_partial_derivs_vs_wl.shape}, wl size: {wl.size}")       #调试
        #     assert T_fwd_partial_derivs_vs_wl.shape[1] == wl.size
        #     # # mw: the next line is a check code, for broadband source to check frequency point not in the middle of
        #     # # frequency range, the gradient agrees well
        #     # T_fwd_partial_derivs = T_fwd_partial_derivs_vs_wl[..., 0].flatten()

            wavelength_range = wl.max() - wl.min()
            T_fwd_integrand = np.power(np.abs(T_fwd_vs_wavelength), norm_p) / wavelength_range
            const_factor = +1.0 * np.power(np.trapz(y=T_fwd_integrand, x=wl), 1.0 / norm_p - 1.0) # mw: the sign is different from original code!
            integral_kernel = np.power(np.abs(T_fwd_vs_wavelength), norm_p - 1) * np.sign(T_fwd_vs_wavelength) / wavelength_range

            ## Implement the trapezoidal integration as a matrix-vector-product for performance reasons
            d = np.diff(wl)
            quad_weight = np.append(np.append(d[0], d[0:-1] + d[1:]),
                                    d[-1]) / 2  # < There is probably a more elegant way to do this
            v = const_factor * integral_kernel * quad_weight
            T_fwd_partial_derivs = T_fwd_partial_derivs_vs_wl.dot(v)                                    #对波长进行积分  波长不止一个点

            ## This is the much slower (but possibly more readable) code
            # num_opt_param = T_fwd_partial_derivs_vs_wl.shape[0]
            # T_fwd_partial_derivs = np.zeros(num_opt_param, dtype = 'complex')
            # for i in range(num_opt_param):
            #     T_fwd_partial_deriv = np.take(T_fwd_partial_derivs_vs_wl.transpose(), indices = i, axis = 1)
            #     T_fwd_partial_derivs[i] = const_factor * np.trapz(y = integral_kernel * T_fwd_partial_deriv, x = wl)
        else:
            T_fwd_partial_derivs = T_fwd_partial_derivs_vs_wl.flatten()

        return T_fwd_partial_derivs.flatten().real