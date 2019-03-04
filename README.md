## Differentiable Programming Tensor Networks

### Ising

Run this to compute the free energy, energy and specific heat of a 2D Ising model using Tensor Renormalization Group + Automatic Differentiation 


```bash
$ python ising.py 
```

### Heisenberg

Run this to optimize an iPEPS wavefuntion for 2D quantum Heisenberg model.


```bash
$ python main.py -D 3 -chi 30 
```

It is also possible to supply your own Hamiltonian, and measure other physical observable at your interests. 

### What is going on ?

The codes in [adlib](https://github.com/wangleiphy/tensorgrad/tree/master/rg/adlib) implements the backward function needed to propagate gradients through tensor network contractions.  

### Requirements

* [PyTorch](https://pytorch.org/)
* A good GPU card  if you are inpatient 