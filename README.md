## Code used to obtain the results in ["Linear cross-entropy certification of quantum computational advantage in Gaussian Boson Sampling"](https://arxiv.org/abs/2403.15339).

* `coeffs_ideal_score.py` computes the different coefficients used to determine the ideal LXE score for given $R$ (number of non-vacuum input modes), $2N$ (total number of detected photons), and input squeezing parameters. This code was used to obtain the results in Figure 4 of the article.
* `sampling_sqs.py` generates samples for a squashed state model with no vacuum input modes, where all the input states have the same mean number of photons.
* `computing_probs_sqs.py` computes the conditional probability of a sample from the squashed state model with respect to an ideal squeezed state model. This code was used to obtain some of the results shown in Figure 6 of the article.
* `sampling_ideal_sqz.py` generates a sample from an ideal squeezed state model and computes its corresponding conditional probability with respect to the same model. This code was used to obtain the results shown in Figures 5 and 6 of the article.
* `sampling_lossy_sqz.py` generates a sample from a lossy squeezed state model and computes its corresponding conditional probability with respect to an ideal squeezed state model. This code was used to obtain the results shown in Figure 6 of the article.
* `requirements.txt` contains all the requirements to run these files.

The ideal model corresponds to a setup with no vacuum input modes, where all the input squeezed states have the same squeezing parameter. The lossy squeezed state model corresponds to a setup with no vacuum input modes, where the input squeezed states (having the same squeezing parameter) are sent through single-mode loss channels with the same transmission parameter before entering the interferometer.

The conditional probabilities have the form $\Pr(\boldsymbol{n}|\boldsymbol{A})/\Pr(2N|\boldsymbol{A})$, where $\boldsymbol{n}$ is a sample, and $\boldsymbol{A}$ is the ideal squeezed state model.
