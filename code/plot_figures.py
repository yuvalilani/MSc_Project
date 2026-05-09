import numpy as np
import matplotlib.pyplot as plt


def save_plot(file_name):
    plt.savefig(file_name + ".svg")
    plt.savefig(file_name + ".png")
    plt.clf()


if __name__ == '__main__':

    mu, gamma_index, rho_index = np.loadtxt("../data/bdi_1d_index_comparison")
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.get_cmap('tab20').colors)
    plt.plot(mu, rho_index, '-', label=r"$\frac{4}{\pi}arg(\mathcal{Z} _{\mathcal{T}})$", linewidth=2)
    plt.plot(mu, gamma_index, '--', label=r"$-\frac{Im(Tr(\Gamma\tilde\Gamma)}{2}$", linewidth=2)
    plt.xlabel(r"$\frac{\mu}{t}$", fontsize=13)
    plt.legend(fontsize="13", loc="upper right")
    plt.grid()
    plt.tight_layout()
    save_plot("../latex/figures/1d BDI index comparison")

    data = np.loadtxt("../data/d_1d_gamma_convergence")
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.get_cmap('tab20').colors)
    plt.plot(data[0], data[1], label="Global Index")
    for j in range(len(data) - 2):
        plt.plot(data[0], data[j + 2], '--', label="Local Index, N=" + str(j + 1))
    plt.xlabel(r"$\mu /t$")
    plt.legend(fontsize="13", loc="upper center")
    plt.grid()
    plt.tight_layout()
    save_plot("../latex/figures/1d D Gamma index convergence")
