from fluidsim.solvers.ns2d.solver import Simul
import numpy as np, math
from netCDF4 import Dataset

dt0=0.004

params = Simul.create_default_params()
params.oper.nx = params.oper.ny = 128
params.oper.Lx = params.oper.Ly = 2.0 * 3.141592653589793
params.time_stepping.t_end = 1000.0
params.time_stepping.deltat0 = dt0
params.time_stepping.USE_CFL = False

digits=abs(math.floor(np.log10(dt0)))

params.forcing.enable = True
params.forcing.type = "in_script"        # fixed pattern you supply
params.forcing.key_forced = "rot_fft"    # force vorticity (standard in ns2d)
params.nu_4 = 4e-6                   # hyperviscosity (example)
params.nu_m4 = 5e-2                   # hypoviscosity (example)

params.output.sub_directory="/home/jwa-user/practice_pytorch/Barotropic_low/output"
params.output.periods_save.phys_fields = 0.4

params.init_fields.type="noise"
params.init_fields.noise.length=0.0

sim = Simul(params)
op = sim.oper

# -----------------------
# 3) hook the forcing time series into 'in_script'
# -----------------------
def adjust_t_each_time(self):
    sim.time_stepping.t = round(sim.time_stepping.t,digits) ### round output time

sim.forcing.forcing_maker.monkeypatch_compute_forcing_fft_each_time(
    compute_forcing_fft_each_time
)

sim.time_stepping.start()
