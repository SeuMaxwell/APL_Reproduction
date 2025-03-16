import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
from matplotlib.animation import FileMovieWriter
from matplotlib.cbook import flatten
import os

class Plotter(object):
    """
    a class to organize the plot for design region, field and gradient
    """
    def __init__(self, save_folder, show_plot=False, save_plot=True):

        self.show_plot = show_plot
        self.save_plot = save_plot
        self.save_folder = save_folder

        if not os.path.exists(self.save_folder):
            os.mkdir(self.save_folder)

    def plot_all(self, optimization, label):
        self.plot_designs(optimization, label)
        self.plot_field(optimization, label)
        self.plot_gradient(optimization, label)


    def plot_designs(self, optimization, label):
        """
        plot the design region permittivity for inspection
        """

        plt.figure()
        plt.title(label)
        ax = plt.gca()
        # sets the ratio to 1
        ax.set_aspect(1)

        optimization.geometry.plot(ax)

        if self.show_plot:
            plt.colorbar()
            plt.show()
        if self.save_plot:
            save_path = os.path.join(self.save_folder, 'design region eps')
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            plt.savefig(os.path.join(save_path, label + ".png"), dpi=200)

        plt.close()

    def plot_field(self, optimization, label):
        """
        plot the field for inspection
        """

        total_plot_num = len(optimization.forward_field_list)
        cmap_list = ['Blues', 'Greens', 'Oranges', 'Reds', ['RdPu']]

        for wl_id in range(total_plot_num):
            plt.figure()
            plt.title(label)
            ax = plt.gca()
            # sets the ratio to 1
            ax.set_aspect(1)

            optimization.forward_field_list[wl_id].plot(ax, title=label + '_wl_{}'.format(wl_id),
                                                        cmap=cmap_list[wl_id])

            if self.show_plot:
                plt.colorbar()
                plt.show()
            if self.save_plot:
                save_path = os.path.join(self.save_folder, 'design region E field')
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                plt.savefig(os.path.join(save_path, label + '_wl_{}.png'.format(wl_id)), dpi=200)

            plt.close()

    def plot_gradient(self, optimization, label):
        """
        plot the field for inspection
        """

        plt.figure()
        plt.title(label)
        ax = plt.gca()
        # sets the ratio to 1
        ax.set_aspect(1)

        optimization.plot_gradient_field(ax, title=label)

        if self.show_plot:
            plt.colorbar()
            plt.show()
        if self.save_plot:
            save_path = os.path.join(self.save_folder, 'design region gradient field')
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            plt.savefig(os.path.join(save_path, label + '.png'), dpi=200)

        plt.close()

