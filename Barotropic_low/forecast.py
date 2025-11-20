import os 
import sys 
import numpy as np, math
from netCDF4 import Dataset
from fluidsim.solvers.ns2d.solver import Simul
from fluidsim.base.output.base import SpecificOutput
import common

nx=128
dt0=0.004
dt_out=0.4
intv_out=int(round(dt_out/dt0))
length=2.0

# Keep a reference to the original method
_orig_has_to_online_save = SpecificOutput._has_to_online_save

# Install the patch globally
SpecificOutput._has_to_online_save = common._has_to_online_save_every_n_steps

params = Simul.create_default_params()
params.oper.nx = params.oper.ny = nx
params.oper.Lx = params.oper.Ly = 2.0 * 3.141592653589793
params.time_stepping.deltat0 = dt0
params.time_stepping.USE_CFL = False

params.output.periods_print["print_stdout"]=10*dt0

digits=abs(math.floor(np.log10(params.time_stepping.deltat0)))

params.forcing.enable = True
params.forcing.type = "in_script"        # fixed pattern you supply
params.forcing.key_forced = "rot_fft"    # force vorticity (standard in ns2d)
params.nu_4 = 4e-6                  # hyperviscosity (example)
params.nu_m4 = 5e-2                   # hypoviscosity (example)

params.output.sub_directory=os.path.join(os.getcwd(),"output")
#params.output.sub_directory="/mnt/work/jbm/amemiya/Barotropic/output"
params.output.periods_save.phys_fields = dt_out
params.output.periods_print["print_stdout"]=10*dt_out

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
path_nc=os.path.join(os.getcwd(),"forcings","forcing.nc")
nc = Dataset(path_nc, "r")

forcing_rot_r=nc["rot_fft_forcing_r"]
forcing_rot_i=nc["rot_fft_forcing_i"]
dt_forcing=nc["time"][1]-nc["time"][0]
nt_forcing=len(nc["time"])
time_intv_forcing=round(dt_forcing/params.time_stepping.deltat0)

# -----------------------
# 3) hook the forcing time series into 'in_script'
# -----------------------
def compute_forcing_fft_each_time(self):

    it=self.sim.time_stepping.it
    it_forcing=min(int(it/time_intv_forcing),nt_forcing-2)
    factor= float(it%time_intv_forcing)/float(time_intv_forcing)
 
    rot_fft_forcing_prev=forcing_rot_r[it_forcing]+1j*forcing_rot_i[it_forcing]
    rot_fft_forcing_next=forcing_rot_r[it_forcing+1]+1j*forcing_rot_i[it_forcing+1]

    rot_fft_forcing=(1.0-factor) * rot_fft_forcing_prev + factor * rot_fft_forcing_next

    return {"rot_fft": rot_fft_forcing}

sim.forcing.forcing_maker.monkeypatch_compute_forcing_fft_each_time(
    compute_forcing_fft_each_time
)

sim.time_stepping.start()
