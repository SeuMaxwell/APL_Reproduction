"""
General filter functions to be used in other projection and morphological transform routines.
图形学变化的一般滤波器函数。
"""

import numpy as np
from autograd import numpy as npa

def _centered(arr, newshape):
    '''Helper function that reformats the padded array of the fft filter operation.
    Borrowed from scipy:
    '''
    # Return the center newshape portion of the array.
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

def _zero_pad(arr, pad):
    # fill sides
    left = npa.tile(0, (pad[0][0], arr.shape[1]))  # left side
    right = npa.tile(0, (pad[0][1], arr.shape[1]))  # right side
    top = npa.tile(0, (arr.shape[0], pad[1][0]))  # top side
    bottom = npa.tile(0, (arr.shape[0], pad[1][1]))  # bottom side

    # fill corners
    top_left = npa.tile(0, (pad[0][0], pad[1][0]))  # top left
    top_right = npa.tile(0, (pad[0][1], pad[1][0]))  # top right
    bottom_left = npa.tile(0, (pad[0][0], pad[1][1]))  # bottom left
    bottom_right = npa.tile(0, (pad[0][1], pad[1][1]))  # bottom right

    out = npa.concatenate((
        npa.concatenate((top_left, top, top_right)),
        npa.concatenate((left, arr, right)),
        npa.concatenate((bottom_left, bottom, bottom_right))
    ), axis=1)

    return out

def _edge_pad(arr, pad):
    # fill sides
    left = npa.tile(arr[0, :], (pad[0][0], 1))  # left side
    right = npa.tile(arr[-1, :], (pad[0][1], 1))  # right side
    top = npa.tile(arr[:, 0], (pad[1][0], 1)).transpose()  # top side
    bottom = npa.tile(arr[:, -1], (pad[1][1], 1)).transpose()  # bottom side)

    # fill corners
    top_left = npa.tile(arr[0, 0], (pad[0][0], pad[1][0]))  # top left
    top_right = npa.tile(arr[-1, 0], (pad[0][1], pad[1][0]))  # top right
    bottom_left = npa.tile(arr[0, -1], (pad[0][0], pad[1][1]))  # bottom left
    bottom_right = npa.tile(arr[-1, -1], (pad[0][1], pad[1][1]))  # bottom right

    out = npa.concatenate((
        npa.concatenate((top_left, top, top_right)),
        npa.concatenate((left, arr, right)),
        npa.concatenate((bottom_left, bottom, bottom_right))
    ), axis=1)

    return out

def simple_2d_filter(x, kernel, Nx, Ny):
    """A simple 2d filter algorithm that is differentiable with autograd.
    Uses a 2D fft approach since it is typically faster and preserves the shape
    of the input and output arrays.
    二维滤波算法   二维快速傅里叶变换方法，通常更快，保留输入和输出数组的形状。
    The ffts pad the operation to prevent any circular convolution garbage.

    Parameters
    ----------
    x : array_like (2D)
        Input array to be filtered. Must be 2D.
    kernel : array_like (2D)
        Filter kernel (before the DFT). Must be same size as `x`
    Nx : int, number of data points along x direction
    Ny : int, number of data points along y direction

    Returns
    -------
    array_like (2D)
        The output of the 2d convolution.
    """
    # Get 2d parameter space shape
    (kx, ky) = kernel.shape

    # Ensure the input is 2D
    x = x.reshape(Nx, Ny)

    # pad the kernel and input to avoid circular convolution and
    # to ensure boundary conditions are met.
    kernel = _zero_pad(kernel, ((kx, kx), (ky, ky)))
    x = _edge_pad(x, ((kx, kx), (ky, ky)))

    # Transform to frequency domain for fast convolution
    H = npa.fft.fft2(kernel)
    X = npa.fft.fft2(x)

    # Convolution (multiplication in frequency domain)
    Y = H * X

    # We need to fftshift since we padded both sides if each dimension of our input and kernel.
    y = npa.fft.fftshift(npa.real(npa.fft.ifft2(Y)))

    # Remove all the extra padding
    y = _centered(y, (kx, ky))

    return y

def conic_filter(x, radius, x_cords, y_cords):
    '''A linear conic filter, also known as a "Hat" filter in the literature [1].

    Parameters
    ----------
    x : array_like (2D)
        Design parameters
    radius : float
        Filter radius (in "meep units")    投影半径
    Nx : int, number of data points along x direction
    Ny : int, number of data points along y direction
    dx : mesh size along x direction
    dy : mesh size along y direction

    Returns
    -------
    array_like (2D)
        Filtered design parameters.

    References
    ----------
    [1] Lazarov, B. S., Wang, F., & Sigmund, O. (2016). Length scale and manufacturability in
    density-based topology optimization. Archive of Applied Mechanics, 86(1-2), 189-218.
    '''

    # Formulate grid over entire design region
    xv, yv = np.meshgrid(x_cords, y_cords, sparse=True, indexing='ij')

    # Calculate kernel
    #判断是否在圆内，是则为   1-半径/距离，   否则为0   赋值作用
    kernel = np.where(np.abs(xv ** 2 + yv ** 2) <= radius ** 2, (1 - np.sqrt(abs(xv ** 2 + yv ** 2)) / radius), 0)


    # Normalize kernel
    kernel = kernel / np.sum(kernel.flatten())  # Normalize the filter

    # Filter the response
    y = simple_2d_filter(x, kernel, len(x_cords), len(y_cords))

    return y

def tanh_projection(x, beta, eta):              #对输入参数进行阈值处理，使其在0和1之间。通常是“最强”的投影。
    '''Projection filter that thresholds the input parameters between 0 and 1. Typically
    the "strongest" projection.

    Parameters
    ----------
    x : array_like
        Design parameters
    beta : float
        Thresholding parameter (0 to infinity). Dictates how "binary" the output will be.
    eta: float
        Threshold point (0 to 1)

    Returns
    -------
    array_like
        Projected and flattened design parameters.
    References
    ----------
    '''

    return (npa.tanh(beta * eta) + npa.tanh(beta * (x - eta))) / (npa.tanh(beta * eta) + npa.tanh(beta * (1 - eta)))

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x = np.arange(0, 1, 0.01)
    eta = 0.5
    beta = 1
    y = tanh_projection(x, beta, eta)

    plt.figure()
    plt.plot(x, y)
    plt.show()
