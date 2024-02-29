==============
About PAL 2.0
==============

The lack of efficient discovery tools for advanced functional materials remains a major bottleneck to enabling advances in the next-generation energy, health, and sustainability technologies. 
One main factor contributing to this inefficiency is the large combinatorial space of materials (with respect to material compositions and processing conditions) that is typically redolent of such materials-centric applications.
Searches of this large combinatorial space are often influenced by expert knowledge and clustered close to material configurations that are known to perform well, thus ignoring potentially high-performing candidates in unanticipated regions of the composition-space or processing protocol. 
Moreover, experimental characterization or first principles quantum mechanical calculations of all possible material candidates can be prohibitively expensive, making exhaustive approaches to determine the best candidates infeasible.
As a result, there remains a need for the development of computational algorithms that can efficiently search a large parameter space for a given material application.
Here, we introduce PAL 2.0, a method that combines a physics-based surrogate model with Bayesian optimization. 
The key contributing factor of our proposed framework is the ability to create a physics-based hypothesis using XGBoost and Neural Networks. 
This hypothesis provides a physics-based ``prior'' (or initial beliefs) to a Gaussian process model, which is then used to perform a search of the material design space.  

Citation
----------------
If you use the databases or code, please cite the paper:

>M. Sharma Priyadarshini, O. Romiluyi, Y. Wang, K. Miskin, C. Ganley and P. Clancy, “PAL 2.0: A Physics-Driven Bayesian Optimization Framework for Material Discovery,” *Mater. Horiz.*, **11**, 781, (2024). DOI: http://doi.org/10.1039/D3MH01474F
