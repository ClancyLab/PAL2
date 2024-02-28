[![Documentation Status](https://readthedocs.org/projects/pal2/badge/?version=latest)](http://pal2.readthedocs.io/)

# PAL 2.0
Welcome to PAL 2.0!

The lack of efficient discovery tools for advanced functional materials remains a major bottleneck to enabling advances in the next-generation energy, health, and sustainability technologies. One main factor contributing to this inefficiency is the large combinatorial space of materials that is typically redolent of such materials-centric applications. Experimental characterization or first principles quantum mechanical calculations of all possible material candidates can be prohibitively expensive, making exhaustive approaches to determine the best candidates infeasible. As a result, there remains a need for the development of computational algorithms that can efficiently search a large parameter space for a given material application. Here, we introduce PAL 2.0, a method that combines a physics-based surrogate model with Bayesian optimization. The key contributing factor of our proposed framework is the ability to create a physics-based hypothesis using XGBoost and Neural Networks. This hypothesis provides a physics-based "prior" (or initial beliefs) to a Gaussian process model, which is then used to perform a search of the material design space.

<hr>

Documentation
----------------

This repository contains code to implement PAL2. It 
was created within the CONDA enviroment, and instructions 
for installing within it are available in the [Documentation](http://pal2.readthedocs.io/).

* Any questions or comments please reach out via email
to the authors of the paper.

<hr>

Code Developers
----------------

The PAL 2.0 code was developed by Maitreyee Sharma Priyadarshini, Seun Romiluyi and Nihaar Thakkar.

Contributors can be found [here](https://github.com/ClancyLab/PAL2/graphs/contributors).

<hr>

Citing
----------------
If you use the databases or code, please cite the paper:

>M. Sharma Priyadarshini, O. Romiluyi, Y. Wang, K. Miskin, C. Ganley and P. Clancy, “PAL 2.0: A Physics-Driven Bayesian Optimization Framework for Material Discovery,” _Mater. Horiz._, **11**, 781, (2024). [(0.1039/D3MH01474F)](http://doi.org/10.1039/D3MH01474F)

<hr>

Acknowledgment
----------------

This work was supported by 
the Department of Energy (DOE), Office of Science, Basic Energy Science (BES), under Award #DE-SC0022305.

<hr>

License
----------------

PAL 2.0 code is distributed under terms of the [MIT License](https://github.com/ClancyLab/PAL2/blob/main/LICENSE.txt).
