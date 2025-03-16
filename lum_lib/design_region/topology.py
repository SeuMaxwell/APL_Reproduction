#怎么实现拓扑结构的优化
from lum_lib.lumerical_simu_api.scripts import set_spatial_interp, get_eps_from_sim
from autograd import tensor_jacobian_product, jacobian, elementwise_grad

import scipy as sp
eps0 = sp.constants.epsilon_0

import numpy as np

import lum_lib.utils.adjoint_utils as adj_utils

def material_scaling(x, device_ep, bg_ep):
    """
    design parameters are from 0-1, scale to target materials
    材料的缩放函数，将设计参数从0-1映射到目标材料的介电常数
    定义设备和背景的介电常数
    """
    return (device_ep - bg_ep) * x + bg_ep


# `material_mapping` 函数的作用是将设计参数 `rho` 进行参数化处理，以构建目标材料的介电常数分布。具体步骤如下：
def material_mapping(rho, device_ep, bg_ep, radius, x_cords, y_cords,
                     beta=1, eta=0.5, if_flatten=False):
    """
    Defines the parameterization steps for constructing rho
    """
    # blur the image with proper kernel
    # 1. **模糊处理**：使用 `adj_utils.conic_filter` 函数对 `rho` 进行模糊处理，以去除小特征和尖锐角。  这个很重要
    # 2. **投影处理**：使用 `adj_utils.tanh_projection` 函数对模糊处理后的 `rho` 进行投影处理，以增强离散性。
    # 3. **材料缩放**：使用 `material_scaling` 函数将投影处理后的 `rho` 映射到目标材料的介电常数范围。
    # 4. **展平处理**：如果 `if_flatten` 为 `True`，则将 `rho` 展平为一维数组。
    # 最终返回处理后的 `rho`。
    rho = adj_utils.conic_filter(x=rho, radius=radius, x_cords=x_cords, y_cords=y_cords)

    # project blurred image
    rho = adj_utils.tanh_projection(rho, beta, eta)

    rho = material_scaling(rho, device_ep, bg_ep)              #材料的缩放函数，将设计参数从0-1映射到目标材料的介电常数

    if if_flatten:
        rho = rho.flatten()

    return rho

# `TopologyOptimization2DParameters`   包括了2D拓扑优化的基本属性和参数，如设计参数、介电常数范围、网格坐标、滤波半径、投影参数等。
# 类定义了二维拓扑优化的参数。具体包括：  二维拓扑优化的参数 计算离散度 更新投影 保存参数到文件 从文件加载参数 计算参数和介电常数之间的关系 从参数计算介电常数 从介电常数计算参数
class TopologyOptimization2DParameters(object):

    def __init__(self, params, eps_min, eps_max, x, y, z, filter_R, eta, beta, beta_factor=2, eps=None):
        self.last_params = params  # 0-1
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.eps = eps
        self.x = x  # array of x coordinates
        self.y = y  # array of y coordinates
        self.z = z
        self.bounds = [(0., 1.)] * (len(x) * len(y))
        self.filter_R = filter_R
        self.eta = eta

        self.beta = beta
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]
        self.dz = z[1] - z[0] if (hasattr(z, "__len__") and len(z) > 1) else 0
        self.depth = z[-1] - z[0] if (hasattr(z, "__len__") and len(z) > 1) else 220e-9
        self.beta_factor = beta_factor
        self.discreteness = 0

        ## Prep-work for unfolding symmetry to properly enforce min feature sizes at symmetry boundaries. Off by default for now!
        self.symmetry_x = False
        self.symmetry_y = False
        self.unfold_symmetry = self.symmetry_x or self.symmetry_y  # < We do want monitors to unfold symmetry but we need to detect if the size does not match the parameters anymore

    def use_interpolation(self):
        return True

    def check_license_requirements(self, sim):
        ## Try to execute one of the topology script commands
        try:
            sim.fdtd.eval(('params = struct;'
                           'params.eps_levels=[1,2];'
                           'params.filter_radius = 3;'
                           'params.beta = 1;'
                           'params.eta = 0.5;'
                           'params.dx = 0.1;'
                           'params.dy = 0.1;'
                           'params.dz = 0.0;'
                           'eps_geo = topoparamstoindex(params,ones(5,5));'))
        except:
            raise UserWarning(
                'Could not execute required topology optimization commands. Either the version of FDTD is outdated or the '
                'license requirements for topology optimization are not fulfilled. Please contact your support or sales representative.')

        return True

    def calc_discreteness(self):
        ''' Computes a measure of discreteness. Is 1 when the structure is completely discrete and less when it is not. '''
        ## Compute the discreteness of the structure 计算结构的离散度   1-4*sum(rho*(1-rho))/N   rho是介电常数  N是介电常数的个数   介电常数是0-1之间的数 介电常数越接近0或1，离散度越高
        rho = self.calc_params_from_eps(self.eps).flatten()
        return 1 - np.sum(4 * rho * (1 - rho)) / len(rho)

    #`update_projection` 函数的作用是更新投影参数并计算当前结构的离散度。具体步骤如下：
    # 计算当前结构的离散度，并将其存储在 `self.discreteness` 中。
    # 打印当前的离散度。
    # 更新投影参数 `beta`，将其乘以 `beta_factor`。
    # 打印更新后的 `beta` 值。
    # 返回当前的离散度和更新后的 `beta` 值。
    def update_projection(self):
        self.discreteness = self.calc_discreteness()
        print("Current Discreteness: {}".format(self.discreteness))

        self.beta *= self.beta_factor
        print('Next Beta is {}'.format(self.beta))

        return self.discreteness, self.beta

    def to_file(self, filename):
        np.savez(filename, params=self.last_params, eps_min=self.eps_min, eps_max=self.eps_max, x=self.x, y=self.y,
                 z=self.z, depth=self.depth, beta=self.beta, eps=self.eps)


    # 1. `calc_params_from_eps(self, eps)`:
    #    - 计算介电常数 `eps` 对应的设计参数 `params`，将介电常数从 `eps_min` 到 `eps_max` 映射到 0 到 1 的范围。
    def calc_params_from_eps(self, eps):
        return np.minimum(np.maximum((eps - self.eps_min) / (self.eps_max - self.eps_min), 0), 1.0)

    # 2. `set_params_from_eps(self, eps)`:
    #    - 设置对象的 `last_params` 属性为由 `eps` 计算得到的设计参数。
    def set_params_from_eps(self, eps):
        self.last_params = self.calc_params_from_eps(eps)

    # 3. `get_eps_from_params(self, params)`:
    #    - 将设计参数 `params` 转换为介电常数 `eps`。
    #    - 重新调整设计参数的形状，并根据对称性扩展或减少设计参数。
    #    - 使用 `material_mapping` 函数将设计参数映射到介电常数。
    #    - 返回计算得到的介电常数 `eps`。
    def get_eps_from_params(self, params):
        rho = np.reshape(params, (len(self.x), len(self.y)))
        self.last_params = rho

        ## Expand symmetry (either along x- or y-direction)
        rho = self.unfold_symmetry_if_applicable(rho)

        ## Extend boundary to include effects from fixed structure
        eps = material_mapping(rho, device_ep=self.eps_min, bg_ep=self.eps_max, radius=self.filter_R,
                      x_cords=self.x, y_cords=self.y, beta=self.beta, eta=self.eta)

        ## Reduce symmetry again (move to cad eventually?)
        if self.symmetry_x:
            shape = rho.shape
            eps = eps[int((shape[0] - 1) / 2):, :]  # np.vstack( (np.flipud(rho)[:-1,:],rho) )
        if self.symmetry_y:
            shape = rho.shape
            eps = eps[:, int((shape[1] - 1) / 2):]  # np.hstack( (np.fliplr(rho)[:,:-1],rho) )

        return eps

    def initialize(self, wavelengths, opt):
        self.opt = opt
        pass

    def update_geometry(self, params):
        self.eps = self.get_eps_from_params(params)
        self.discreteness = self.calc_discreteness()

    def unfold_symmetry_if_applicable(self, rho):
        ## Expand symmetry (either along x- or y-direction)
        if self.symmetry_x:
            rho = np.vstack((np.flipud(rho)[:-1, :], rho))
        if self.symmetry_y:
            rho = np.hstack((np.fliplr(rho)[:, :-1], rho))
        return rho

    def get_current_params_inshape(self, unfold_symmetry=False):
        return self.last_params

    def get_current_eps(self):
        return self.eps

    def get_current_params(self):
        params = self.get_current_params_inshape()
        return np.reshape(params, (-1)) if params is not None else None

    def plot(self, ax_eps):
        ax_eps.clear()
        x = self.x * 1e6
        y = self.y * 1e6
        eps = self.eps

        ax_eps.pcolormesh(x, y, np.real(np.transpose(eps)), vmin=self.eps_min, vmax=self.eps_max, cmap='Greys',
                          shading='auto')

        ax_eps.set_title('Eps')
        ax_eps.set_xlabel('x(um)')
        ax_eps.set_ylabel('y(um)')

#继承自 `TopologyOptimization2DParameters` 类，定义了一个二维拓扑优化区域。具体包括：  从文件加载参数  设置参数  计算梯度  添加几何  从文件加载参数  设置参数  计算梯度  添加几何
class TopologyOptimization2D(TopologyOptimization2DParameters):
    '''
    '''
    def __init__(self, params, eps_min, eps_max, x, y, z=0, filter_R=200e-9, eta=0.5, beta=1, beta_factor=2, eps=None,
                 min_feature_size=0):
        super().__init__(params, eps_min, eps_max, x, y, z, filter_R, eta, beta, beta_factor, eps)

    @classmethod
    def from_file(cls, filename, z=0, filter_R=200e-9, eta=0.5, beta=None):
        data = np.load(filename)
        if beta is None:
            beta = data["beta"]
        return cls(data["params"], data["eps_min"], data["eps_max"], data["x"], data["y"], z=z, filter_R=filter_R,
                   eta=eta, beta=beta, eps=data["eps"])

    def set_params_from_eps(self, eps):
        # Use the permittivity in z-direction. Does not really matter since this is just used for the initial guess and is (usually) heavily smoothed
        super().set_params_from_eps(eps[:, :, 0, 0, 2])

    def calculate_gradients(self, gradient_fields):

        # rho = self.get_current_params_inshape()
        rho = self.get_current_params()

        # If we have frequency data (3rd dim), we need to adjust the dimensions of epsilon for broadcasting to work
        # field E has the shape: (151, 151, 1, 11, 3) (Nx, Ny, Nz, wavelength_num, 3)
        E_forward_dot_E_adjoint = np.atleast_3d(
            np.real(np.squeeze(np.sum(gradient_fields.get_field_product_E_forward_adjoint(), axis=-1))))

        # the gradient is the Re(E_adj * E_old), dot product
        # eps0 * dV term is put here for numerical reasons, check Modematch.get_adjoint_field_scaling()
        dF_dEps = 2 * eps0 * E_forward_dot_E_adjoint * self.dx * self.dy

        project_func = lambda x: material_mapping(x, device_ep=self.eps_min, bg_ep=self.eps_max, radius=self.filter_R,
                      x_cords=self.x, y_cords=self.y, beta=self.beta, eta=self.eta,
                      if_flatten=True)

        num_of_freq = dF_dEps.shape[-1]
        if num_of_freq == 1:
            topo_grad = tensor_jacobian_product(project_func, 0)(rho, dF_dEps.flatten())
        else:
            # for multi-frequency gradient, do it one by one
            topo_grad_list = list()
            for i in range(num_of_freq):
                topo_grad_list.append(tensor_jacobian_product(project_func, 0)(rho, dF_dEps[..., i].flatten()))
            topo_grad = np.stack(topo_grad_list, axis=-1)

        return topo_grad

    def add_geo(self, sim, params=None, only_update=False):

        fdtd = sim.fdtd

        eps = self.eps if params is None else self.get_eps_from_params(params.reshape(-1))

        fdtd.putv('x_geo', self.x)
        fdtd.putv('y_geo', self.y)
        fdtd.putv('z_geo', np.array([self.z - self.depth / 2, self.z + self.depth / 2]))

        if not only_update:
            set_spatial_interp(sim.fdtd, 'opt_fields', 'specified position')
            set_spatial_interp(sim.fdtd, 'opt_fields_index', 'specified position')

            script = ('select("opt_fields");'
                      'set("x min",{});'
                      'set("x max",{});'
                      'set("y min",{});'
                      'set("y max",{});').format(np.amin(self.x), np.amax(self.x), np.amin(self.y), np.amax(self.y))
            fdtd.eval(script)

            script = ('select("opt_fields_index");'
                      'set("x min",{});'
                      'set("x max",{});'
                      'set("y min",{});'
                      'set("y max",{});').format(np.amin(self.x), np.amax(self.x), np.amin(self.y), np.amax(self.y))
            fdtd.eval(script)

            script = ('addimport;'
                      'set("detail",1);')
            fdtd.eval(script)

            mesh_script = ('addmesh;'
                           'set("x min",{});'
                           'set("x max",{});'
                           'set("y min",{});'
                           'set("y max",{});'
                           'set("dx",{});'
                           'set("dy",{});').format(np.amin(self.x), np.amax(self.x), np.amin(self.y), np.amax(self.y),
                                                   self.dx, self.dy)
            fdtd.eval(mesh_script)

        if eps is not None:
            fdtd.putv('eps_geo', eps)

            ## We delete and re-add the import to avoid a warning
            script = ('select("import");'
                      'delete;'
                      'addimport;'
                      'temp=zeros(length(x_geo),length(y_geo),2);'
                      'temp(:,:,1)=eps_geo;'
                      'temp(:,:,2)=eps_geo;'
                      'importnk2(sqrt(temp),x_geo,y_geo,z_geo);')
            fdtd.eval(script)


## 类继承2D的参数，重写了一些方法，主要是将3D的epsilon转换为2D的epsilon    添加计算梯度calculate_gradients  添加几何add_geo
class TopologyOptimization3DLayered(TopologyOptimization2DParameters):

    def __init__(self, params, eps_min, eps_max, x, y, z, filter_R=200e-9, eta=0.5, beta=1, beta_factor=2, eps=None,
                 min_feature_size=0):
        super(TopologyOptimization3DLayered, self).__init__(params, eps_min, eps_max, x, y, z, filter_R, eta, beta,
                                                            beta_factor, eps)

    @classmethod
    def from_file(cls, filename, filter_R, eta=0.5, beta=None):
        data = np.load(filename)
        if beta is None:
            beta = data["beta"]
        return cls(data["params"], data["eps_min"], data["eps_max"], data["x"], data["y"], data["z"], filter_R=filter_R,
                   eta=eta, beta=beta)

    def to_file(self, filename):
        np.savez(filename, params=self.last_params, eps_min=self.eps_min, eps_max=self.eps_max, x=self.x, y=self.y,
                 z=self.z, beta=self.beta, eps=self.eps)

    def set_params_from_eps(self, eps):
        '''
            The raw epsilon of a 3d system needs to be collapsed to 2d first. For now, we just pick the first z-layer
        '''
        midZ_idx = int((eps.shape[2] + 1) / 2)
        super().set_params_from_eps(eps[:, :, midZ_idx, 0, 2])

    def calculate_gradients(self, gradient_fields):

        rho = self.get_current_params()

        # If we have frequency data (3rd dim), we need to adjust the dimensions of epsilon for broadcasting to work
        # field E has the shape: (151, 151, 1, 11, 3) (Nx, Ny, Nz, wavelength_num, 3)
        E_forward_dot_E_adjoint = np.real(
            np.squeeze(np.sum(gradient_fields.get_field_product_E_forward_adjoint(), axis=-1)))

        ## We integrate/sum along the z-direction,
        E_forward_dot_E_adjoint_int_z = np.atleast_3d(np.squeeze(np.sum(E_forward_dot_E_adjoint, axis=2)))

        # the gradient is the Re(E_adj * E_old), dot product
        # eps0 * dV term is put here for numerical reasons, check Modematch.get_adjoint_field_scaling()
        dF_dEps = 2 * eps0 * E_forward_dot_E_adjoint_int_z * self.dx * self.dy * self.dz



        project_func = lambda x: material_mapping(x, device_ep=self.eps_min, bg_ep=self.eps_max, radius=self.filter_R,
                                                  x_cords=self.x, y_cords=self.y, beta=self.beta, eta=self.eta,
                                                  if_flatten=True)

        num_of_freq = dF_dEps.shape[-1]
        print("rho shape:", rho.shape)
        print("dF_dEps original shape:", dF_dEps.shape)
        print("dF_dEps flattened shape:", dF_dEps.flatten().shape)
        # 在topology.py中的calculate_gradients函数中:
        if dF_dEps.size > rho.size:
            # 去除边缘的额外点
            dF_dEps = dF_dEps[1:-1, :, :]  # 从403x86x1裁剪到401x86x1
        assert rho.size == dF_dEps.size, f"维度不匹配: rho({rho.size}) vs dF_dEps({dF_dEps.size})"
        if num_of_freq == 1:
            topo_grad = tensor_jacobian_product(project_func, 0)(rho, dF_dEps.flatten())

        else:
            # for multi-frequency gradient, do it one by one
            topo_grad_list = list()
            for i in range(num_of_freq):
                topo_grad_list.append(tensor_jacobian_product(project_func, 0)(rho, dF_dEps[..., i].flatten()))
            topo_grad = np.stack(topo_grad_list, axis=-1)
            #     # 修复：确保topo_grad是一个具有正确维度的数组 (params_count, num_of_freq)
            #     # 而不只是将它们堆叠到新的轴上
            # topo_grad = np.column_stack(topo_grad_list)

        return topo_grad

    def add_geo(self, sim, params=None, only_update=False):

        fdtd = sim.fdtd

        eps = self.eps if params is None else self.get_eps_from_params(params.reshape(-1))

        if not only_update:
            set_spatial_interp(sim.fdtd, 'opt_fields', 'specified position')
            set_spatial_interp(sim.fdtd, 'opt_fields_index', 'specified position')

            script = ('select("opt_fields");'
                      'set("x min",{});'
                      'set("x max",{});'
                      'set("y min",{});'
                      'set("y max",{});'
                      'set("z min",{});'
                      'set("z max",{});').format(np.amin(self.x), np.amax(self.x), np.amin(self.y), np.amax(self.y),
                                                 np.amin(self.z), np.amax(self.z))
            fdtd.eval(script)

            script = ('select("opt_fields_index");'
                      'set("x min",{});'
                      'set("x max",{});'
                      'set("y min",{});'
                      'set("y max",{});'
                      'set("z min",{});'
                      'set("z max",{});').format(np.amin(self.x), np.amax(self.x), np.amin(self.y), np.amax(self.y),
                                                 np.amin(self.z), np.amax(self.z))
            fdtd.eval(script)

            script = ('addimport;'
                      'set("detail",1);')
            fdtd.eval(script)

            mesh_script = ('addmesh;'
                           'set("x min",{});'
                           'set("x max",{});'
                           'set("y min",{});'
                           'set("y max",{});'
                           'set("z min",{});'
                           'set("z max",{});'
                           'set("dx",{});'
                           'set("dy",{});'
                           'set("dz",{});').format(np.amin(self.x), np.amax(self.x), np.amin(self.y), np.amax(self.y),
                                                   np.amin(self.z), np.amax(self.z), self.dx, self.dy, self.dz)
            fdtd.eval(mesh_script)

        if eps is not None:
            # This is a layer geometry, so we need to expand it to all layers
            full_eps = np.broadcast_to(eps[:, :, None], (len(self.x), len(self.y), len(self.z)))

            fdtd.putv('x_geo', self.x)
            fdtd.putv('y_geo', self.y)
            fdtd.putv('z_geo', self.z)
            fdtd.putv('eps_geo', full_eps)

            ## We delete and re-add the import to avoid a warning
            script = ('select("import");'
                      'delete;'
                      'addimport;'
                      'importnk2(sqrt(eps_geo),x_geo,y_geo,z_geo);')
            fdtd.eval(script)
