import os, sys, glob
from fluidsim.solvers.ns2d.solver import Simul
import numpy as np, math
from netCDF4 import Dataset

dt0=0.002

params = Simul.create_default_params()
params.oper.nx = params.oper.ny = 256
params.oper.Lx = params.oper.Ly = 2.0 * 3.141592653589793
params.time_stepping.t_end = 10.0
params.time_stepping.deltat0 = dt0
params.time_stepping.USE_CFL = False

params.output.periods_print["print_stdout"]=dt0

digits=abs(math.floor(np.log10(params.time_stepping.deltat0)))

params.forcing.enable = True
params.forcing.type = "in_script"        # fixed pattern you supply
params.forcing.key_forced = "rot_fft"    # force vorticity (standard in ns2d)
params.nu_4 = 1e-8                   # hyperviscosity (example)
params.nu_m4 = 5e-2                   # hypoviscosity (example)

params.output.sub_directory="/home/jwa-user/practice_pytorch/Barotropic/output"
params.output.periods_save.phys_fields = dt0

params.init_fields.type="from_file"

init_nc=sys.argv[1]
params.init_fields.from_file.path=init_nc
nc=Dataset(init_nc,"r")
init_time = nc["state_phys"].time

lead_time=10

params.time_stepping.t_end = init_time + lead_time


sim = Simul(params)
op = sim.oper

# --- open a NetCDF file to load the forcing ---
path_nc = "/home/jwa-user/practice_pytorch/Barotropic/forcings_new/forcing.nc"
nc = Dataset(path_nc, "r")

forcing_rot_r=nc["rot_fft_forcing_r"]
forcing_rot_i=nc["rot_fft_forcing_i"]
dt_forcing=nc["time"][1]-nc["time"][0]
nt_forcing=len(nc["time"])
time_intv_forcing=round(dt_forcing/params.time_stepping.deltat0)

# --- open a NetCDF file to log the forcing ---
path_nc = sim.output.path_run + "/forcing_additonal.nc"
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
it_forcing=0
time_intv_forcing_out=1

# --- wrap the time-stepping function ---
orig_step = sim.time_stepping.one_time_step_computation

# --- define the callback executed every time step ---
def one_step_and_save():
    global _it
    global it_forcing
    global time_intv_forcing_out
    orig_step()  # perform the normal time step
    Fh = sim.forcing.get_forcing()   # spectral forcing (complex array)
    Fp = op.ifft(Fh)             # physical forcing (complex array)

#    print(Fh.shape)
#    print(Fp.shape)
#    quit()
    sim.time_stepping.t = round(sim.time_stepping.t,digits)

    if np.mod(_it,time_intv_forcing_out) == 0: 
        vtime[it_forcing] = sim.time_stepping.t
        vrot[it_forcing, :, :] = Fp.real.astype("f4")
        vrot_fft_r[it_forcing, :, :] = Fh.real.astype("f4")
        vrot_fft_i[it_forcing, :, :] = Fh.imag.astype("f4")
        nc.sync()
        it_forcing += 1

    _it += 1

# replace the integrator's single-step method
sim.time_stepping.one_time_step_computation = one_step_and_save
## add callback to run every step (same frequency as deltat0)
#sim.time_stepping.add_callback_every_dt(save_forcing_each_step,
#                                        dt=params.time_stepping.deltat0)






# -----------------------
# 3) hook the forcing time series into 'in_script'
# -----------------------
def compute_forcing_fft_each_time(self):
###    sim.time_stepping.t = round(sim.time_stepping.t,digits) ### round output time

    it=self.sim.time_stepping.it
    it_forcing=min(int(it/time_intv_forcing),nt_forcing-2)
    factor= float(it%time_intv_forcing)/float(time_intv_forcing)

#    print("t,it,it_forcing",self.sim.time_stepping.t,it,it_forcing)
 
    rot_fft_forcing_prev=forcing_rot_r[it_forcing]+1j*forcing_rot_i[it_forcing]
    rot_fft_forcing_next=forcing_rot_r[it_forcing+1]+1j*forcing_rot_i[it_forcing+1]

    rot_fft_forcing=(1.0-factor) * rot_fft_forcing_prev + factor * rot_fft_forcing_next

    return {"rot_fft": rot_fft_forcing}

sim.forcing.forcing_maker.monkeypatch_compute_forcing_fft_each_time(
    compute_forcing_fft_each_time
)

sim.time_stepping.start()
