from fluidsim.solvers.ns2d.solver import Simul
import numpy as np, math
from netCDF4 import Dataset

params = Simul.create_default_params()
params.oper.nx = params.oper.ny = 256
params.oper.Lx = params.oper.Ly = 2.0 * 3.141592653589793
params.time_stepping.t_end = 10.0
params.time_stepping.deltat0 = 0.002
params.time_stepping.USE_CFL = False

params.forcing.enable = True
#params.forcing.type = "in_script"        # fixed pattern you supply
#params.forcing.key_forced = "rot_fft"    # force vorticity (standard in ns2d)
params.forcing.type = "tcrandom"      # e.g. time-correlated random ring forcing
params.forcing.nkmin_forcing = 4
params.forcing.nkmax_forcing = 5
params.forcing.normalized.constant_rate_of = "energy"
params.forcing.forcing_rate = 0.2        # set target energy input rate
params.nu_4 = 1e-8                   # hyperviscosity (example)
params.nu_m4 = 5e-2                   # hypoviscosity (example)

#
#print(params.forcing)
#quit()

params.output.sub_directory="/home/jwa-user/practice_pytorch/Barotropic/output"
params.output.periods_save.phys_fields = 2.0

params.init_fields.type="noise"
params.init_fields.noise.length=0.0
sim = Simul(params)
op = sim.oper

# --- open a NetCDF file to log the forcing ---
path_nc = sim.output.path_run + "/forcing.nc"
nc = Dataset(path_nc, "w")
nc.createDimension("time", None)
nc.createDimension("y", op.ny)
nc.createDimension("x", op.nx)
nc.createDimension("l", op.ny)
nc.createDimension("k", op.nx/2+1)
vtime = nc.createVariable("time", "f8", ("time",))
vrot  = nc.createVariable("rot_forcing", "f4", ("time","y","x"), zlib=True)
vrot_fft_r  = nc.createVariable("rot_fft_forcing_r", "f4", ("time","l","k"), zlib=True)
vrot_fft_i  = nc.createVariable("rot_fft_forcing_i", "f4", ("time","l","k"), zlib=True)

_it = 0

# --- wrap the time-stepping function ---
orig_step = sim.time_stepping.one_time_step_computation

# --- define the callback executed every time step ---
def one_step_and_save():
    global _it
    orig_step()  # perform the normal time step
    Fh = sim.forcing.get_forcing()   # spectral forcing (complex array)
    Fp = op.ifft(Fh)             # physical forcing (complex array)

#    print(Fh.shape)
#    print(Fp.shape)
#    quit()

    vtime[_it] = sim.time_stepping.t
    vrot[_it, :, :] = Fp.real.astype("f4")
    vrot_fft_r[_it, :, :] = Fh.real.astype("f4")
    vrot_fft_i[_it, :, :] = Fh.imag.astype("f4")
    nc.sync()
    _it += 1

# replace the integrator's single-step method
sim.time_stepping.one_time_step_computation = one_step_and_save
## add callback to run every step (same frequency as deltat0)
#sim.time_stepping.add_callback_every_dt(save_forcing_each_step,
#                                        dt=params.time_stepping.deltat0)

sim.time_stepping.start()
