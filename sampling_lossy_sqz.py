"""
    Generates a sample from a lossy squeezed state model 
    and computes its conditional probability with respect to the ideal model.
"""

import argparse
import numpy as np
from scipy.special import factorial, binom
from thewalrus import hafnian_repeated
from thewalrus.samples import generate_hafnian_sample
from thewalrus.symplectic import passive_transformation


def uniform_lossy_sqz_cov(num_modes, squeezing_par, trans_loss, hbar=2):
    """Computes the covariance matrix of num_modes lossy squeezed states
    with the same squeezing and transmission loss parameters

    Args:
        num_modes (int): number of modes
        squeezing_par (float): squeezing parameter
        trans_loss (float): transmission loss parameter
        hbar (float): value of the constant in the commutation relation [x,p] = i * hbar

    Returns:
        (array): real covariance matrix of the multimode lossy squeezed state

    """
    x_diag_elems = [trans_loss * np.exp(-2 * squeezing_par) + 1 - trans_loss] * num_modes
    p_diag_elems = [trans_loss * np.exp(2 * squeezing_par) + 1 - trans_loss] * num_modes
    diagonal_elems = x_diag_elems + p_diag_elems
    return (hbar / 2) * np.diag(diagonal_elems)


def ideal_sqz_cond_prob(sample, U_mat):
    """Computes the conditional probability of obtaining a sample with a given number of photons
    according to the ideal squeezed sate model

    Args:
        sample (array): photon number detection pattern
        U_mat (array): unitary matrix representing the interferometer

    Returns:
        (float): probability of obtaining the sample according to the ideal squeezed sate model
    """
    num_modes = U_mat.shape[0]
    num_photons = sample.sum()
    V_mat = U_mat @ U_mat.T
    tr_haf = np.abs(hafnian_repeated(V_mat, sample)) ** 2
    tr_fac = factorial(sample).prod()
    tr_bin = binom(num_modes / 2 + num_photons / 2 - 1, num_photons / 2)
    return tr_haf / (tr_fac * tr_bin)


parser = argparse.ArgumentParser(
    description="Generate a sample from the lossy squeezed state model distribution"
)
parser.add_argument("--M", type=int, help="Number of modes")
parser.add_argument("--I", type=int, help="ID of the unitary")
parser.add_argument("--S", type=int, help="ID of the generated sample")
parser.add_argument("--eta", type=float, help="Transmission coefficient")
args = parser.parse_args()

total_mean_num_photons = 20
sqz_par = np.arcsinh(np.sqrt(total_mean_num_photons / args.M))

haar_U = np.load(f"./random_unitaries/unitaries_squeezed/U_sqz_M_{args.M}_id_{args.I}.npy")
sqz_cov = uniform_lossy_sqz_cov(num_modes=args.M, squeezing_par=sqz_par, trans_loss=args.eta)
_, evol_sqz_cov = passive_transformation(mu=np.zeros(2 * args.M), cov=sqz_cov, T=haar_U)

det_pattern = generate_hafnian_sample(evol_sqz_cov, max_photons=30, cutoff=10)
prob = ideal_sqz_cond_prob(np.array(det_pattern), haar_U)

print(args.S, prob, *det_pattern)
