import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def covariance_matrix(n, mu=0.0):
    t = 1.0
    delta = 1.0
    a = np.zeros((2 * n, 2 * n))
    for i in range(n - 1):
        j = (i + 1) % n
        a[2 * i + 1, 2 * j] = t + delta
        a[2 * i, 2 * j + 1] = delta - t
        a[2 * i, 2 * i + 1] = mu
    a = a - a.T
    w, v = np.linalg.eigh(1j * a)
    return np.imag(v @ np.diag(np.sign(w)) @ v.conj().T)


def partial_transpose(gamma, a: list):
    trs = np.identity(len(gamma), dtype=np.complex128)
    for i in range(a[1], a[2]):
        trs[2 * i, 2 * i] *= 1j
        trs[2 * i + 1, 2 * i + 1] *= -1j
    return trs @ gamma @ trs


def matrix_flux(mat_a, mat_b, a: list):
    total_flux = 0.0
    for j in range(a[0] * 2, a[1] * 2):
        for k in range(a[1] * 2, a[2] * 2):
            total_flux += mat_a[j, k] * mat_b[j, k]
    return total_flux


def local_probe(mat_a, mat_b, a):
    m_a = mat_a[2*a[0]:2*a[2], 2*a[0]:2*a[2]]
    m_b = mat_b[2*a[0]:2*a[2], 2*a[0]:2*a[2]]
    return -np.imag(np.trace(m_a @ m_b))/2


def ground_state_projector(gamma):
    proj = (np.diag(np.ones(len(gamma))) - 1j * gamma) / 2
    return proj


def density_matrix(gamma, gamma_t, a: list):
    proj = ground_state_projector(gamma)
    proj_t = ground_state_projector(gamma_t)
    a_mat = np.zeros_like(gamma)
    for i in range(2 * a[0], 2 * a[2]):
        a_mat[i, i] = 1
    def func(p):
        return (a_mat @ p @ a_mat +
               (np.identity(len(gamma)) - a_mat) @ p @ (np.identity(len(gamma)) - a_mat))
    m = np.identity(len(gamma)) - proj + proj @ func(proj_t) @ proj
    return m


def compare_topological_indices(n=200, region_size=0.5):
    n_cut = n // 2
    a = [int(n_cut - region_size/2 * n), n_cut, int(n_cut + region_size/2 * n)]

    mu = np.linspace(-4, 4, 101)
    gamma_index = np.zeros(len(mu), dtype=np.complex128)
    rho_index = np.zeros(len(mu), dtype=np.complex128)
    for i in tqdm(range(len(mu))):
        gamma = covariance_matrix(n, mu[i])
        gamma_t = partial_transpose(gamma, a)
        gamma_index[i] = local_probe(gamma, gamma_t, a)
        mat = density_matrix(gamma, gamma_t, a)
        rho_index[i] = np.angle(np.linalg.det(mat)) / np.pi * 4

    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.get_cmap('tab20').colors)
    plt.plot(mu, rho_index, '-', label=r"$\frac{4}{\pi}arg(\mathcal{Z} _{\mathcal{T}})$", linewidth=2)
    plt.plot(mu, gamma_index, '--', label=r"$-\frac{Im(Tr(\Gamma\tilde\Gamma)}{2}$", linewidth=2)
    plt.xlabel(r"$\frac{\mu}{t}$", fontsize=13)
    plt.legend(fontsize="13", loc="upper right")
    plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    compare_topological_indices(n = 100)
