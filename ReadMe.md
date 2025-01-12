# **Target-Aware Ratio Importance Sampling (TARIS)**

This repository provides the code accompanying my master thesis. It is divided into two subfolders: 

- **`toy_example`**: Contains a toy example utilizing the adaptive MCMC implementation of TARIS.  
- **`amortized_toy_example`**: Contains the amortized implementation using a Normalizing Flow to approximate the optimal TARIS proposal.

---

## **Environment Setup**

To set up the environment, run the following commands:

```bash
conda env create -f environment.yaml
conda activate msc-thesis
pip install -e . --no-deps --no-build-isolation
```

---

## **Toy Example**

This folder provides the code accompanying the first toy example of my master thesis. It includes an implementation of the adaptive MCMC version of TARIS and a comparison to the TABI estimator.  

- The `TARIS.py` file contains all functions necessary to compute both TABI and TARIS for this specific experiment.

---

## **Amortized Toy Example**

This folder provides the code for an amortized toy example of my master thesis. The experiment setup is as follows:

```math
\begin{aligned}
   p(x) & = \mathcal{N}(x; 5, 1), \\ 
   p(y \mid x) & = \mathcal{N}(y; x, 1), \\ 
   f(x) & = \min(15000, \max(0, 50(x - 4)^5)).
\end{aligned}
```

We use the non-iterative alternative optimal proposal:  
```math
q_{\text{opt}}^a \propto p(x, y) \sqrt{f(x)},
```  
to compare amortized TARIS and amortized TABI in an early-stopping scenario for robustness. To achieve the best results for amortized TABI, Golinski et al. train for 500 epochs. We have reduced training to 50 epochs, but the performance difference can already be observed after the first epoch.

---

## **Instructions**

### 1. Train the Proposal
Train `q_ratio` by running:
```bash
python msc_thesis/amortized_toy_example/train_non_iterative_new.py
```
After each epoch a model checkpoint will be saved in the output folder.

### 2. Generate Samples
To generate samples from the learned proposal, which will be combined into estimates in the next step, run:
```bash
python msc_thesis/amortized_toy_example/generate_samples_non_iterative_new.py msc_thesis/amortized_toy_example/config_1d_non_iterative.yaml
```

This script saves the generated samples in the output folder.

### 3. Compute ReMSE Figures
To reproduce the ReMSE figures:  
- Use the `estimator_new_experiments.ipynb` notebook alongside the `estimators.py` file.  
- Run the **AMCI** estimator on the problem by adapting the `tail_integral_1d` setup to this problem:  
  - Update \( f(x) \) in the `model.py` file.  
  - Ensure it does not amortize over \( \theta \) by modifying the sampling process to return \( \theta = 4 \).

---

## **Miscellaneous**

A big thank you to the developers and contributors of:  
- [AMCI](https://github.com/talesa/amci)  
- [PyTorch Flows](https://github.com/ikostrikov/pytorch-flows)  
- [Higher](https://github.com/facebookresearch/higher)

If you have any questions, encounter issues, or need clarification, please create an issue in the repository. I'll try to get back to you promptly.

