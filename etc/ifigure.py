import matplotlib.pyplot as plt

class Ifigure():
    def __init__(self, ncols = 2, nrows = 2):
        fig, axs = plt.subplots(ncols=ncols, nrows=nrows)
        self.fig = None
        self.axs = None
        self.ncols = ncols
        self.nrows = nrows

    def draw(self, mfunc, ax, **kwargs):
        if ax is None:
            ax = plt.gca()  # use current axis if none is given
        plt.sca(ax)
        mfunc(kwargs)
        ax.grid(False)

    def visualize(self, axs = None):

        if len(axs) > (self.nrows * self.ncols):
            self.nrows = int(len(axs)/self.ncols)+1

        if self.fig is None:
            fig, fig = plt.subplots(ncols=self.ncols, nrows=self.nrows)
            self.fig = fig
            self.axs = fig



        plt.ion()


        plot1, = axs[0, 0].plot(np.log10(np.abs(noise_average_score)), 'r', label='Error in Noise Model')
        plot2, = axs[0, 0].plot(np.log10(np.abs(signal_average_scroe)), 'b', label='Error in Signal Model')
        if iter == 0:
            axs[0, 0].legend(loc='upper right', bbox_to_anchor=(0, 0.5))

        axs[0, 1].clear()
        plot4, = axs[0, 1].plot(df_noise[features].values, df_noise['y'], c='r', alpha=0.5, linestyle='None',
                                marker='o', label='noise')
        plot3, = axs[0, 1].plot(df_signal[features].values, df_signal['y'], c='b', alpha=0.5, linestyle='None',
                                marker='o', label='signal')
        plot3_1, = axs[0, 1].plot(df_move[features].values, df_move['y'], c='k', linestyle='None',
                                  marker='.', label='signal')

        plot5, = axs[1, 0].plot(frac_noise_list, c='k')

        axs[0, 1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plot1.set_ydata(np.log10(np.abs(noise_average_score)))
        plot2.set_ydata(np.log10(np.abs(signal_average_scroe)))
        #
        # plot1.set_ydata(np.log10(np.abs(noise_average_score)))
        # plot2.set_ydata(np.log10(np.abs(signal_average_scroe)))

        plot3.set_ydata(df_signal['y'])
        plot4.set_ydata(df_noise['y'])
        plot3_1.set_ydata(df_move['y'])

        plot3.set_xdata(df_signal[features])
        plot4.set_xdata(df_noise[features])
        plot3_1.set_xdata(df_move[features])

        plot5.set_ydata(frac_noise_list)

        axs[0, 0].set_title(str(iter))
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.1)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    dash = Ifigure()
    fig1 = plt.plot([1,2,3])
    fig2 = plt.plot([1, 2, 5])
    dash.visualize(axs = [fig1, fig2])



