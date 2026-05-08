import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from numba import njit
from pfapack.pfaffian import pfaffian


class HoneycombLattice:

    def __init__(self, lattice_len, x_flip=False, y_flip=False):
        self.lattice_len = lattice_len
        self.lattice_size = 2 * self.lattice_len ** 2
        self.lattice_indices = np.zeros((self.lattice_len, self.lattice_len, 2), dtype=int)
        self.lattice_positions = np.zeros((self.lattice_size, 3), dtype=int)
        self.lattice_neighbors = -np.ones((self.lattice_size, 3), dtype=int)

        # dirac fermion matrices
        self.n_mat = np.zeros((2 * self.lattice_size, 2 * self.lattice_size))
        self.nu_mat = np.zeros((2 * self.lattice_size, 2 * self.lattice_size))
        self.xi_mat = np.zeros((2 * self.lattice_size, 2 * self.lattice_size))
        self.r_mat = np.zeros((2 * self.lattice_size, 2 * self.lattice_size), dtype=np.complex128)

        # flatten the 2d lattice to 1d
        counter = 0
        for i in range(np.shape(self.lattice_indices)[0]):
            for j in range(np.shape(self.lattice_indices)[1]):
                self.lattice_indices[i, j, 0] = counter
                self.lattice_positions[counter] = np.array([i, j, 0])
                self.lattice_indices[i, j, 1] = counter + 1
                self.lattice_positions[counter + 1] = np.array([i, j, 1])
                counter += 2

        # find the neighbors of each index
        for i in range(self.lattice_size):
            if self.lattice_positions[i, 2] == 0:
                j = self.lattice_indices[self.lattice_positions[i, 0], self.lattice_positions[i, 1], 1]
                self.lattice_neighbors[i, 0] = j
                self.lattice_neighbors[j, 0] = i

                if self.lattice_positions[i, 0] < lattice_len - 1 or not y_flip:
                    j = self.lattice_indices[(self.lattice_positions[i, 0] + 1) % lattice_len, self.lattice_positions[i, 1], 1]
                else:
                    j = self.lattice_indices[(self.lattice_positions[i, 0] + 1) % lattice_len, lattice_len - 1 - self.lattice_positions[i, 1], 1]
                self.lattice_neighbors[i, 1] = j
                self.lattice_neighbors[j, 1] = i

                if self.lattice_positions[i, 1] < lattice_len - 1 or not x_flip:
                    j = self.lattice_indices[self.lattice_positions[i, 0], (self.lattice_positions[i, 1] + 1) % lattice_len, 1]
                else:
                    j = self.lattice_indices[lattice_len - 1 - self.lattice_positions[i, 0], (self.lattice_positions[i, 1] + 1) % lattice_len, 1]
                self.lattice_neighbors[i, 2] = j
                self.lattice_neighbors[j, 2] = i

        # build Hamiltonian matrices
        for i in range(self.lattice_size):
            s = 1 - 2 * self.lattice_positions[i, 2]
            self.xi_mat[2 * i, 2 * i] = s
            self.xi_mat[2 * i + 1, 2 * i + 1] = s
            for j in range(3):
                if self.lattice_neighbors[i, j] != -1:
                    n = self.lattice_neighbors[i, j]
                    self.n_mat[2 * i, 2 * n] = 1.0
                    self.n_mat[2 * i + 1, 2 * n + 1] = 1.0
                    for k in range(2):
                        j_n = (j + k + 1) % 3
                        if self.lattice_neighbors[n, j_n] != -1:
                            self.nu_mat[2 * i, 2 * self.lattice_neighbors[n, j_n]] = 1 - 2 * k
                            self.nu_mat[2 * i + 1, 2 * self.lattice_neighbors[n, j_n] + 1] = -(1 - 2 * k)
                    d = self.vertex_position(n) - self.vertex_position(i)
                    d /= np.linalg.norm(d)
                    self.r_mat[2 * i, 2 * n + 1] = d[1] - 1j * d[0]
                    self.r_mat[2 * i + 1, 2 * n] = d[1] + 1j * d[0]

    def dirac_correlation_matrix(self, params):
        # build Hamiltonian matrix
        a = (params[0] * self.n_mat + 1j * params[1] * self.nu_mat + params[2] * self.xi_mat +
             1j * params[3] * self.r_mat)

        # find the ground state
        eigenvalues, eigenvectors = np.linalg.eigh(a)
        u = eigenvectors[:, :len(eigenvalues) // 2]
        p = u @ u.conj().transpose()
        return p

    def vertex_position(self, ind):
        v = np.array(
            [1 / np.sqrt(12) * (1 - 2 * self.lattice_positions[ind, 2]) - (self.lattice_len - 1) * np.sqrt(3) / 2, 0.0])
        v += self.lattice_positions[ind, 0] * np.array([np.sqrt(0.75), 0.5])
        v += self.lattice_positions[ind, 1] * np.array([np.sqrt(0.75), -0.5])
        return v

    def plot_lattice(self):
        for i in range(self.lattice_size):
            v = self.vertex_position(i)
            if self.lattice_positions[i, 2] == 0:
                plt.plot(v[0], v[1], 'bo')
            else:
                plt.plot(v[0], v[1], 'ro')

    def set_regions(self, r, center=np.array([0.0, 0.0])):
        region_indices = [[], [], []]
        for i in range(self.lattice_size):
            v = self.vertex_position(i) - center
            if v[0] ** 2 + v[1] ** 2 < r ** 2:
                ind = math.floor(3 * math.atan2(v[1], v[0]) / (2 * np.pi))
                region_indices[ind].append(i)
        for i in range(3):
            region_indices[i] = np.array(region_indices[i])
        return region_indices

    def get_flake(self, x_range, y_range):
        ind_arr = np.zeros((x_range[1] - x_range[0], y_range[1] - y_range[0], 2), dtype=int)
        for i in range(x_range[0], x_range[1]):
            for j in range(y_range[0], y_range[1]):
                ind_arr[i - x_range[0], j - y_range[0], 0] = self.lattice_indices[i, j, 0]
                ind_arr[i - x_range[0], j - y_range[0], 1] = self.lattice_indices[i, j, 1]
        return ind_arr


@njit
def calculate_topological_index(regions, p):
    new_regions = (np.zeros(len(regions[0]) * 2, dtype=np.int16),
                   np.zeros(len(regions[1]) * 2, dtype=np.int16),
                   np.zeros(len(regions[2]) * 2, dtype=np.int16))
    for i in range(3):
        for j in range(len(regions[i])):
            new_regions[i][2 * j] = 2 * regions[i][j]
            new_regions[i][2 * j + 1] = 2 * regions[i][j] + 1

    nu_topo = 0.0
    for i in new_regions[0]:
        for j in new_regions[1]:
            for k in new_regions[2]:
                s = (-1) ** (i + j + k)
                nu_topo += (p[i, j] * p[j, k] * p[k, i]) * s
                # nu_topo += (p[i, j] * p[j, k] * p[k, i])
                # nu_topo -= (p[i, k] * p[k, j] * p[j, i])

    return -np.real(nu_topo * 12j * np.pi)


@njit
def quaternion_chern_number(regions, p):
    q_mat = np.zeros((2, 2), dtype=np.complex128)
    for i in regions[0]:
        for j in regions[1]:
            for k in regions[2]:
                q_mat += p[2 * i:2 * i + 2, 2 * j:2 * j + 2] @ p[2 * j:2 * j + 2, 2 * k:2 * k + 2] @ p[
                    2 * k:2 * k + 2, 2 * i:2 * i + 2]
                q_mat -= p[2 * k:2 * k + 2, 2 * j:2 * j + 2] @ p[2 * j:2 * j + 2, 2 * i:2 * i + 2] @ p[
                    2 * i:2 * i + 2, 2 * k:2 * k + 2]
    return np.imag(q_mat[0, 0]) * 12 * np.pi


def overlap_flake(p, lat: HoneycombLattice, x_range, y_range, edge_width):
    p_closed = np.zeros_like(p)
    p_closed += p
    flake = lat.get_flake(x_range, y_range)

    for i in flake.flatten():
        # x edge
        for j in range(x_range[0] - edge_width, x_range[1]):
            j_eff = (j - x_range[0]) % (x_range[1] - x_range[0]) + x_range[0]
            for k in range(edge_width):
                for l in range(2):
                    old_index = lat.lattice_indices[j, k + y_range[1], l]
                    new_index = lat.lattice_indices[j_eff, k + y_range[0], l]
                    p_closed[2 * i:2 * i + 2, 2 * new_index: 2 * new_index + 2] += p[
                        2 * i:2 * i + 2, 2 * old_index: 2 * old_index + 2]
                    p_closed[2 * new_index: 2 * new_index + 2, 2 * i:2 * i + 2] += p[
                        2 * old_index: 2 * old_index + 2, 2 * i:2 * i + 2]

        # y edge
        for k in range(y_range[0], y_range[1] + edge_width):
            k_eff = (k - y_range[0]) % (y_range[1] - y_range[0]) + y_range[0]
            for j in range(edge_width):
                for l in range(2):
                    old_index = lat.lattice_indices[j + x_range[1], k, l]
                    new_index = lat.lattice_indices[j + x_range[0], k_eff, l]
                    p_closed[2 * i:2 * i + 2, 2 * new_index: 2 * new_index + 2] += p[
                        2 * i:2 * i + 2, 2 * old_index: 2 * old_index + 2]
                    p_closed[2 * new_index: 2 * new_index + 2, 2 * i:2 * i + 2] += p[
                        2 * old_index: 2 * old_index + 2, 2 * i:2 * i + 2]

    p_closed = p_closed[np.ix_(flake.flatten(), flake.flatten())]
    return p_closed, flake


if __name__ == '__main__':
    lat = HoneycombLattice(10, x_flip=False, y_flip=False)
    lat_1 = HoneycombLattice(10, x_flip=False, y_flip=True)
    lat_2 = HoneycombLattice(10, x_flip=True, y_flip=True)
    lat_3 = HoneycombLattice(10, x_flip=True, y_flip=False)
    # lat.plot_lattice()
    # flake = lat.get_flake([4, 10], [4, 10])
    # flake = lat.get_flake([0, 10], [0, 10])
    # ind = [i for i in range(2 * lat.lattice_size) if i // 2 in flake.flatten()]
    # for i in flake.flatten():
    #     v = lat.vertex_position(i)
    #     plt.plot(v[0], v[1], 'y.')
    # edge_1 = lat.get_flake([1, 10], [10, 13])
    # edge_2 = lat.get_flake([10, 13], [4, 13])
    # for i in edge_1.flatten():
    #     v = lat.vertex_position(i)
    #     plt.plot(v[0], v[1], 'g*')
    # for i in edge_2.flatten():
    #     v = lat.vertex_position(i)
    #     plt.plot(v[0], v[1], 'w*')

    # x = np.linspace(0.0, 10.0, 21)
    # # x = np.unique(np.concatenate((np.linspace(0.0, 10.0, 21), np.linspace(4.0, 6.0, 41))))
    # y = np.zeros(len(x), dtype=np.complex128)
    # rashba = 0.0
    # for i in tqdm(range(len(x))):
    #     p = lat.dirac_correlation_matrix([1.0, 1.0, x[i], rashba])
    #     u_t = np.zeros((len(p), len(p)))
    #     for j in range(len(p) // 2):
    #         u_t[2 * j, 2 * j + 1] = 1
    #         u_t[2 * j + 1, 2 * j] = -1
    #     a = u_t @ (2 * p - np.identity(len(p)))
    #     # a = (a - a.T) / 2
    #     y[i] = pfaffian(a)
    # #     y[i] = quaternion_chern_number(r, p)
    # # x, y = np.loadtxt("2d_data l=20")
    # plt.plot(x, np.real(y), '-o')
    # plt.plot(x, np.imag(y), '-o')
    # plt.plot([np.sqrt(27 - 2.25 * rashba ** 2), np.sqrt(27 - 2.25 * rashba ** 2)], [-1, 1], '--k')
    # plt.xlabel(r"$\lambda_\nu/\lambda_{SO}$")
    # # plt.legend()
    # plt.grid()

    p = lat.dirac_correlation_matrix([1.0, 1.0, 0.0, 0.0])
    p_1 = lat_1.dirac_correlation_matrix([1.0, 1.0, 0.0, 0.0])
    p_2 = lat_2.dirac_correlation_matrix([1.0, 1.0, 0.0, 0.0])
    p_3 = lat_3.dirac_correlation_matrix([1.0, 1.0, 0.0, 0.0])

    e, v = np.linalg.eigh(p)
    e, v_1 = np.linalg.eigh(p_1)
    e, v_2 = np.linalg.eigh(p_2)
    e, v_3 = np.linalg.eigh(p_3)

    print(np.linalg.det(v[:, len(e)//2:].conj().T @ v_1[:, len(e)//2:]))
    print(np.linalg.det(v_1[:, len(e)//2:].conj().T @ v_2[:, len(e)//2:]))
    print(np.linalg.det(v_2[:, len(e)//2:].conj().T @ v_3[:, len(e)//2:]))
    print(np.linalg.det(v_3[:, len(e)//2:].conj().T @ v[:, len(e)//2:]))

    plt.show()
