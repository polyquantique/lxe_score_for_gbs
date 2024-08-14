## Code used to obtain the results in ["Linear cross-entropy certification of quantum computational advantage in Gaussian Boson Sampling"](https://arxiv.org/abs/2403.15339).

* `coeffs_ideal_score.py` computes the different coefficients used to determine the ideal LXE score for given R (number of non-vacuum input modes), 2N (total number of detected photons), and input squeezing parameters. This code was used to obtain the results in Figure 4 of the article.
* `sampling_sqs.py` generates samples for a squashed state model with no vacuum input modes, where all the input states have the same mean number of photons.
* `computing_probs_sqs.py` computes the probability of a sample form the squashed state model with respect to the distribution of an ideal squeezed state model. The ideal model corresponds to a setup with no vacuum input modes, where all the input squeezed states have the same squeezing parameter. This code was used to obtain some of the results shown in Figure 6 of the article.
*    
