"""Generates samples from a squashed state model."""

import argparse
import numpy as np
from thewalrus.symplectic import passive_transformation
from thewalrus.samples import hafnian_sample_classical_state


def uniform_sqs_cov(num_modes, mean_num_photons, hbar=2):
    """Computes covariance matrix of num_modes squashed states with the same mean number of photons

    Args:
        num_modes (int): number of modes
        mean_num_photons (float): mean number of photons
        hbar (float): value of the constant in the commutation relation [x,p] = i * hbar

    Returns:
        (array): real covariance matrix of the multimode squashed state

    """
    x_diag_elems = [1] * num_modes
    p_diag_elems = [1 + 4 * mean_num_photons] * num_modes
    diagonal_elems = x_diag_elems + p_diag_elems
    return (hbar / 2) * np.diag(diagonal_elems)


parser = argparse.ArgumentParser(description="Generate samples form the squashed state model.")
parser.add_argument("--I", type=int, help="ID of the unitary")
parser.add_argument("--num_samples", type=int, help="Number of samples")
args = parser.parse_args()

total_mean_num_photons = 20
haar_U = np.load(f"./random_unitaries/unitaries_squashed/U_sqs_M_200_id_{args.I}.npy")

sqs_cov = uniform_sqs_cov(
    haar_U.shape[0], mean_num_photons=total_mean_num_photons / haar_U.shape[0]
)
_, evol_sqs_cov = passive_transformation(np.zeros(len(sqs_cov)), sqs_cov, haar_U)
samples = hafnian_sample_classical_state(
    evol_sqs_cov, args.num_samples, mean=np.zeros(len(sqs_cov))
)
np.save(f"./all_samples_sqs_id_{args.I}", samples)
