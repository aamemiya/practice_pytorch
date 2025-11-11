from fluidsim.solvers.ns2d.solver import Simul
import numpy as np, math

params = Simul.create_default_params()
params.oper.nx = params.oper.ny = 256
params.time_stepping.t_end = 100.0
params.time_stepping.it_end = 10

params.forcing.enable = True
params.forcing.type = "in_script"        # fixed pattern you supply
params.forcing.key_forced = "rot_fft"    # force vorticity (standard in ns2d)
params.forcing.normalized.constant_rate_of = "energy"
params.forcing.forcing_rate = 1.0        # set target energy input rate

params.output.sub_directory="/home/jwa-user/practice_pytorch/Barotropic/output"
params.output.periods_save.phys_fields = 10.0

params.init_fields.type="noise"
params.init_fields.noise.length=0.0
sim = Simul(params)

#print(params.init_fields)
#quit()

# Define your spatial pattern once (e.g., Kolmogorov f ~ sin(k y))
Y = sim.oper.Y; ly = sim.oper.ly; k = 6*math.pi/ly
X = sim.oper.X; lx = sim.oper.lx; l = 8*math.pi/lx
forcing0 = np.sin(k*Y) * np.cos(l*X)                  # fixed in space

def compute_forcingc_each_time(self):
    return forcing0                      # time-constant pattern
sim.forcing.forcing_maker.monkeypatch_compute_forcing_each_time(
    compute_forcingc_each_time
)

sim.time_stepping.start()
