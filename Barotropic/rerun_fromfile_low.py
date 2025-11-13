from fluidsim.solvers.ns2d.solver import Simul
import numpy as np, math
from netCDF4 import Dataset

params = Simul.create_default_params()
params.oper.nx = params.oper.ny = 128
params.oper.Lx = params.oper.Ly = 2.0 * 3.141592653589793
params.time_stepping.t_end = 10.0
params.time_stepping.deltat0 = 0.002
params.time_stepping.USE_CFL = False

params.forcing.enable = True
params.forcing.type = "in_script"        # fixed pattern you supply
params.forcing.key_forced = "rot_fft"    # force vorticity (standard in ns2d)
params.nu_4 = 1e-6                   # hyperviscosity (example)
params.nu_m4 = 5e-2                   # hypoviscosity (example)

params.output.sub_directory="/home/jwa-user/practice_pytorch/Barotropic/output_low"
params.output.periods_save.phys_fields = 2.0

params.init_fields.type="from_file"
params.init_fields.from_file.path="/home/jwa-user/practice_pytorch/Barotropic/dummy_low_init/state_phys_t0008.002.nc"

sim = Simul(params)
op = sim.oper

# --- open a NetCDF file to load the forcing ---
path_nc = "/home/jwa-user/practice_pytorch/Barotropic/forcing_test_low.nc"
nc = Dataset(path_nc, "r")

forcing_rot_r=nc["rot_fft_forcing_r"]
forcing_rot_i=nc["rot_fft_forcing_i"]
dt_forcing=nc["time"][1]-nc["time"][0]
print(nc["time"][0:6])
if dt_forcing != params.time_stepping.deltat0 :
    print("oh no is time step mismatch",dt_forcing,params.time_stepping.deltat0)
    quit()
# -----------------------
# 3) hook the forcing time series into 'in_script'
# -----------------------
def compute_forcing_fft_each_time(self):
    t = self.sim.time_stepping.t
    # nearest-neighbor in time (or do linear interpolation; see B below)
    n = int(round(t/params.time_stepping.deltat0))-1
    rot_fft_forcing=forcing_rot_r[n]+1j*forcing_rot_i[n]
    return {"rot_fft": rot_fft_forcing}

sim.forcing.forcing_maker.monkeypatch_compute_forcing_fft_each_time(
    compute_forcing_fft_each_time
)


sim.time_stepping.start()
