# Amortized Toy Example

This repo provides the code accompanying the amortized toy example of my master thesis.


## Environment setup
conda create -n amci
Conda activate amci
Conda install pip
Pip install torch torchvision
conda install --file requirements_conda.txt
pip install -r requirements_pip.txt

## Instructions 

0. `cd` into the repo
0. Train `q_ratio` by running 
    ```
    python train_non_iterative_new.py

    ``` 
0. To generate samples from the learned proposal that will be combined into estimates in the next step run
    ```
    python generate_samples_non_iterative_new.py config_1d_non_iterative.yaml
    ```
    The script saves the generated samples onto the hard drive.
0. To reproduce the ReMSE figures run the estimator_new_experiments.ipynb notebook using the estimators.py file. Also run AMCI on this problem. This is easiest done by adapting the 'tail_integral_1d' problem to our problem by simply changing $f(x)$ in the `model.py` file.
    

## Miscellaneous
Many thanks to the developers and contributors of [github.com/talesa/amci](https://github.com/talesa/amci), [github.com/ikostrikov/pytorch-flows](https://github.com/ikostrikov/pytorch-flows/) and [github.com/facebookresearch/higher](https://github.com/facebookresearch/higher/).

If anything is unclear, doesn't work, or you just have questions please create an issue, I'll try to get back to you.


