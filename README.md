## Differentiable Programming Tensor Networks

### Taking gradient of free energy

Run this to compute the energy and specific heat of a 2D classical Ising model by Automatic Differentiation through the Tensor Renormalization Group.


```bash
$ python ising.py 
```

### Variational optimization of iPEPS

Run this to optimize an iPEPS wavefuntion for 2D quantum Heisenberg model. Here, we use Corner Transfer Matrix Renormalization Group for contraction, and L-BFGS for optimization. 


```bash
$ python main.py -D 3 -chi 30 
```

It is also possible to supply your own Hamiltonian and measure other physical observable of interests. 

### What is going on ?

Reverse mode AD computes gradient accurately and efficiently for you! Check the codes in [adlib](https://github.com/wangleiphy/tensorgrad/tree/master/rg/adlib) for backward functions which propagate gradients through tensor network contractions.  

### Requirements

* [PyTorch 1.0+](https://pytorch.org/)
* A good GPU card if you are inpatient or ambitious 