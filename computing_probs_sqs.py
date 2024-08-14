"""
    Computes the conditional probability of a sample from the squashed state model 
    with respect to the ideal model.
"""

import argparse
import numpy as np
from scipy.special import factorial, binom
from thewalrus import hafnian_repeated


def ideal_sqz_cond_prob(sample, U_mat):
    """Computes the conditional probability of obtaining a sample with a given number of photons
    according to the ideal squeezed sate model

    Args:
        sample (array): photon number detection pattern
        U (array): unitary matrix representing the interferometer

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
    description="Compute conditional probabilities of squashed state samples."
)
parser.add_argument("--I", type=int, help="ID of the unitary")
parser.add_argument("--N", type=int, help="Total number of detected photons")
parser.add_argument("--S", type=int, help="ID of the sample")
args = parser.parse_args()

haar_U = np.load(f"./random_unitaries/unitaries_squashed/U_sqs_M_200_id_{args.I}.npy")
det_pattern = np.load(f"./samples/samples_squashed/samples_sqs_M_200_2N_{args.N}_id_{args.I}.npy")[
    args.S
]
prob = ideal_sqz_cond_prob(det_pattern, haar_U)

print(args.S, prob)
