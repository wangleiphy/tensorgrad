import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument("-folder", default='../data/',help="where to store results")
parser.add_argument("-d", type=int, default=2, help="d")
parser.add_argument("-D", type=int, default=2, help="D")
parser.add_argument("-chi", type=int, default=20, help="chi")
parser.add_argument("-chi_obs", type=int, default=None, help="chi for measure observables")
parser.add_argument("-Nepochs", type=int, default=100, help="Nepochs")
parser.add_argument("-Niter", type=int, default=100, help="Niter")
parser.add_argument("-Jz", type=float, default=1.0, help="Jz")
parser.add_argument("-Jxy", type=float, default=1.0, help="Jxy")
parser.add_argument("-hx", type=float, default=1.0, help="hx")
parser.add_argument("-model", default='Heisenberg', choices=['TFIM', 'Heisenberg'], help="model name")
parser.add_argument("-load", default=None, help="load wavefunction file")
parser.add_argument("-save_period", type=int, default=1, help="")
parser.add_argument("-float32", action='store_true', help="use float32")
parser.add_argument("-use_checkpoint", action='store_true', help="use checkpoint")
parser.add_argument("-cuda", type=int, default=-1, help="GPU #")

args = parser.parse_args()

