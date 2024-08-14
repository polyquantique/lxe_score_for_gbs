"""Computes the coefficients needed to determine the ideal LXE score."""

import argparse
from itertools import product
import numpy as np
from scipy.special import factorial
from sympy.combinatorics import Permutation
from sympy.combinatorics.named_groups import SymmetricGroup
from sympy.sets import FiniteSet
from sympy.utilities.iterables import partitions
from graph import Graph


def th_moment(sqz_par, a):
    """Computes the a-th 'moment' of tanh(sqz_par), defined in Eq.(72) of the article.

    Args:
        sqz_par (array): Array of input squeezing parameters.
        a (int): order of the moment.

    Returns:
        (float): a-th moment of tanh(sqz_par).

    """
    th_a = np.tanh(sqz_par) ** a
    return th_a.mean()


def bsinv_group(degree):
    """Computes all the permutations that leave the index set [1,...,n]+[1,...,n] invariant.

    Args:
        degree (int): Degree of the permutation group, 2n.

    Returns:
        (list): List of permutations that leave [1,...,n]+[1,...,n] invariant.

    """
    n = degree // 2
    rset = FiniteSet(*list(range(n)))
    pgroup = []
    for sset in rset.powerset():
        cycles = []
        for k in sset:
            cycles.append([k, k + n])
        cycles = Permutation(cycles, size=degree)
        pgroup.append(cycles)
    return pgroup


def omega_perm(kvec):
    """Computes the Omega permutation. See Eq.(14) of the article.

    Args:
        kvec (array): Array of non-negative integers satisfying k[0] + ... + n * k[n-1] = n.

    Returns:
        (sympy permutation): Omega permutation.

    """
    perm_cycle = []
    start_index = 0
    for k in range(len(kvec)):
        if kvec[k] == 0:
            continue
        length = 2 * (k + 1)
        for r in range(kvec[k]):
            first_index = start_index + r * length
            last_index = first_index + length
            perm_cycle.append(list(range(first_index, last_index)))
        start_index = last_index
    return Permutation(perm_cycle)


def restricted_partitions(n):
    """Computes all n-tuples of non-negative integers satisfying k[0] + ... + n * k[n-1] = n.

    Args:
        n (int): Length of the tuple. It is also the integer to be partitioned.

    Returns:
        (array): Array of tuples satisfying k[0] + ... + n * k[n-1] = n.

    """
    all_tuples = []
    for p in partitions(n):
        n_tuple = np.array([int(p.get(k + 1) or 0) for k in range(n)])
        all_tuples.append(n_tuple)
    return np.array(all_tuples)


def coset_type(perm):
    """Computes the coset-type of a permutation (the degree of the permutation must be even).

    Args:
        perm (sympy permutation): Input permutation.

    Returns:
        (array): Coset-type of perm.

    """
    n = perm.size
    indices = [k + 1 for k in range(n)]
    perm_indices = perm(indices)
    edges_1 = [(indices[2 * k], indices[2 * k + 1]) for k in range(n // 2)]
    edges_2 = [(perm_indices[2 * k], perm_indices[2 * k + 1]) for k in range(n // 2)]
    perm_graph = Graph(from_list=edges_1 + edges_2)
    coset_ty = sorted([len(l) // 2 for l in perm_graph.components()], reverse=True)
    return np.array(coset_ty)


def perm_direct_sum(perm_1, perm_2):
    """Computes the direct sum of two permutations.

    Args:
        perm_1 (sympy permutation): First permutation.
        perm_2 (sympy permutation): Second permutation.

    Returns:
        (sympy permutation): direct sum of perm_1 and perm_2.

    """
    index_1 = perm_1.array_form
    index_2 = [jnx + len(index_1) for jnx in perm_2.array_form]
    res_index = index_1 + index_2
    return Permutation(res_index)


parser = argparse.ArgumentParser(description="Compute coefficients for the lxe score")
parser.add_argument("--n", type=int, help="Photon number sector")
parser.add_argument("--R", type=int, help="Number of filled input modes")
args = parser.parse_args()

num_photons = args.n
inv_group = bsinv_group(2 * num_photons)
symm_group = SymmetricGroup(num_photons).elements
res_partitions = restricted_partitions(num_photons // 2)

filled_modes = args.R
rng = np.random.default_rng(40148)
sqz_par = np.array([a + rng.random(filled_modes) for a in [0, 0.33, 0.66, 0.99]])

den_coeff = np.zeros((4, num_photons // 2))
num_coeff = np.zeros((5, num_photons))

for kvec in res_partitions:
    dk_1 = factorial(kvec).prod()
    dk_2 = np.array([(2 * (a + 1)) ** kvec[a] for a in range(num_photons // 2)]).prod()
    for k in range(4):
        nk = np.array(
            [th_moment(sqz_par[k, :], 2 * (a + 1)) ** kvec[a] for a in range(num_photons // 2)]
        ).prod()
        den_coeff[k, kvec.sum() - 1] += nk / (dk_1 * dk_2)

for kvec, lvec in product(res_partitions, res_partitions):
    omega_k = omega_perm(kvec)
    omega_l = omega_perm(lvec)
    dk_1 = factorial(kvec).prod()
    dk_2 = np.array([(2 * (a + 1)) ** kvec[a] for a in range(num_photons // 2)]).prod()
    dl_1 = factorial(lvec).prod()
    dl_2 = np.array([(2 * (a + 1)) ** lvec[a] for a in range(num_photons // 2)]).prod()
    for sigma, est in product(symm_group, inv_group):
        sigma_1 = perm_direct_sum(Permutation(num_photons - 1), sigma)
        tau = (sigma_1 ** (-1)) * est * sigma_1
        tau_kl = tau * perm_direct_sum(omega_k, omega_l)
        coset_type_tau = coset_type(tau_kl)
        coset_length_tau = len(coset_type_tau)
        for k in range(5):
            if k == 4:
                num_coeff[k, coset_length_tau - 1] += (1 / (dk_1 * dk_2)) * (1 / (dl_1 * dl_2))
            else:
                nk = np.array([th_moment(sqz_par[k, :], 2 * b) for b in coset_type_tau]).prod()
                num_coeff[k, coset_length_tau - 1] += (1 / (dk_1 * dk_2)) * (1 / (dl_1 * dl_2)) * nk

np.save(f"./c_coeff_R_{filled_modes}_2N_{num_photons}", num_coeff)
np.save(f"./d_coeff_R_{filled_modes}_2N_{num_photons}", den_coeff)
np.save(f"./sqz_par_R_{filled_modes}".format(filled_modes), sqz_par)
