## Differentiable Programming Tensor Networks


### Requirements

* [PyTorch 1.0+](https://pytorch.org/)
* A good GPU card if you are inpatient or ambitious 

### Higher order gradient of free energy

Run this to compute the energy and specific heat of a 2D classical Ising model by Automatic Differentiation through the Tensor Renormalization Group.


```bash
$ cd 1_ising_TRG
$ python ising.py 
```

You can supply the command line argument `-use_checkpoint` to reduce the memory usage. 

### Variational optimization of iPEPS

Run this to optimize an iPEPS wavefuntion for 2D quantum Heisenberg model. Here, we use Corner Transfer Matrix Renormalization Group for contraction, and L-BFGS for optimization. 


```bash
$ cd 2_variational_iPEPS
$ python variational.py -D 3 -chi 30 
```

In case of a question, you can type `python variational.py -h`. To make use GPU, you can add `-cuda <GPUID>`.  It is also possible to supply Hamiltonian of your own interests. 

### What is under the hood ?

Reverse mode AD computes gradient accurately and efficiently for you! Check the codes in [adlib](https://github.com/wangleiphy/tensorgrad/tree/master/tensornets/adlib) for backward functions which propagate gradients through tensor network contractions.  



