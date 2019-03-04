## Differentiable Programming Tensor Networks

### Appetizer

The code [ising.py](https://github.com/wangleiphy/tensorgrad/tree/master/ising.py) computes the free energy, energy and specific heat of a 2D Ising model using Tensor Renormalization Group + Automatic Differentiation. 

Type this to run 


```bash
$ python ising.py 
```

### Main dish 

Run this, 


```bash
$ python main.py -D 3 -chi 30 
```

which will optimize an iPEPS for 2D quantum Heisenberg model. It is straightforward to supply your own Hamiltonian. 

### Desert 

[power.py](https://github.com/wangleiphy/tensorgrad/tree/master/rg/adlib/power.py) contains an example of customized backward through dominant eigensolver. 

### What is going on ?

The codes in [adlib](https://github.com/wangleiphy/tensorgrad/tree/master/rg/adlab) implements the backward function needed to compute gradients for a tensor network contraction.  

### Requirements

* [PyTorch](https://pytorch.org/)
* A good GPU card in case you are inpatient 