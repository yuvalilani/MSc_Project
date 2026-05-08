import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from pfapack.pfaffian import pfaffian
from sympy import bell


def covariance_matrix(n, mu=0.0, d_phase=np.pi / 2, mu_disorder=[], anti_pbc=False):
    t = 1.0
    delta = t * np.exp(1j * d_phase)
    a = np.zeros((2 * n, 2 * n))
    for i in range(n):
        j = (i + 1) % n
        if j == 0 and anti_pbc:
            sgn = -1
        else:
            sgn = 1
        a[2 * i + 1, 2 * j] = (t + np.real(delta)) * sgn
        a[2 * i, 2 * j + 1] = (-t + np.real(delta)) * sgn
        a[2 * i, 2 * j] = np.imag(delta) * sgn
        a[2 * i + 1, 2 * j + 1] = -np.imag(delta) * sgn
        a[2 * i, 2 * i + 1] = mu
        if len(mu_disorder) > 0:
            a[2 * i, 2 * i + 1] += mu_disorder[i]
    a = (a - a.T) / 2
    w, v = np.linalg.eigh(1j * a)
    return np.imag(v @ np.diag(np.sign(w)) @ v.conj().T)


def partial_transpose(gamma, a: list):
    trs = np.identity(len(gamma))
    for i in range(a[1], a[2]):
        trs[2 * i, 2 * i] *= -1
        trs[2 * i + 1, 2 * i + 1] *= -1
    return trs @ gamma @ trs


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


def local_probe(gamma, gamma_t, a, order: int):
    k = (-gamma @ gamma_t - np.identity(len(gamma_t)))[2 * a[0]: 2 * a[2], 2 * a[0]: 2 * a[2]]
    c_k = np.zeros(order)
    for i in range(order):
        c_k[i] = (-1)**i / (2 * (i + 1)) * np.trace(np.linalg.matrix_power(k, i + 1)) * math.factorial(i + 1)
    s = 1.0
    for i in range(order):
        b_n = 0.0
        for j in range(i + 1):
            b_n += float(bell(i + 1, j + 1, c_k))
        s += b_n / math.factorial(i + 1)
    return s


if __name__ == '__main__':
    n = 100
    n_cut = n // 2
    region_size = 0.5
    a = [int(n_cut - region_size / 2 * n), n_cut, int(n_cut + region_size / 2 * n)]

    mu = np.linspace(-4, 4, 100)
    mu_disorder = np.random.uniform(-0, 0, n)
    global_index = np.zeros_like(mu)
    local_index = np.zeros_like(mu)
    for i in tqdm(range(len(mu))):
        gamma = covariance_matrix(n, mu[i])
        gamma_a = covariance_matrix(n, mu[i], anti_pbc=True)
        gamma_t = partial_transpose(gamma, [0, a[1], n])
        p = ground_state_projector(gamma)
        p_a = ground_state_projector(gamma_a)
        global_index[i] = pfaffian(gamma) / pfaffian(gamma_a)
        local_index[i] = local_probe(gamma, gamma_t, a, 6)
        local_index[i] = np.real(np.linalg.det(np.identity(len(p)) - p + p @ (2 * p_a - np.identity(len(p_a))) @ p))
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.get_cmap('tab20').colors)
    plt.plot(mu, global_index, label="Global Index")
    plt.plot(mu, local_index, '--', label="Local Index")
    plt.xlabel(r"$\mu /t$")
    plt.legend(fontsize="13", loc="upper center")
    plt.grid()
    plt.show()
