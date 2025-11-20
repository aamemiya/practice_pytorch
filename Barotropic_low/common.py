import os 
import numpy as np, math
from netCDF4 import Dataset
from fluidsim.solvers.ns2d.solver import Simul
from fluidsim.base.output.base import SpecificOutput

class ForcingData :
    def __init__(self,path_nc):
        nc = Dataset(path_nc, "r")
        self.frot_r=nc["rot_fft_forcing_r"]
        self.frot_i=nc["rot_fft_forcing_i"]
        self.dt_forcing=nc["time"][1]-nc["time"][0]
        self.nt_forcing=len(nc["time"])
# -----------------------
# 3) hook the forcing time series into 'in_script'
# -----------------------
    def compute_forcing_fft_each_time():

        time_intv_forcing=round(self.dt_forcing/self.sim.time_stepping.deltat)
        it=self.sim.time_stepping.it
        it_forcing=min(int(it/time_intv_forcing),nt_forcing-2)
        factor= float(it%time_intv_forcing)/float(time_intv_forcing)

        ffrot_prev=self.frot_r[it_forcing]+1j*self.frot_i[it_forcing]
        ffrot_next=self.frot_r[it_forcing+1]+1j*self.frot_i[it_forcing+1]

        ffrot=  (1.0-factor) * ffrot_prev + factor * ffrot_next

        return {"rot_fft": ffrot}




def _has_to_online_save_every_n_steps(self):
    """Return True every N iterations instead of using time buckets.

    N is computed once as round(period_save / dt), assuming (nearly) constant dt.
    """
    # If saving is disabled for this output, never save
    if self.period_save == 0:
        return False

    sim = self.sim
    it = sim.time_stepping.it

    # Lazy initialization the first time this is called for a given SpecificOutput
    if not hasattr(self, "_n_save_steps"):
        dt = sim.time_stepping.deltat

        if dt <= 0.0:
            # Fallback: save every step if something odd happens
            self._n_save_steps = 1
        else:
            # N ≈ period_save / dt, rounded to nearest integer, at least 1
            self._n_save_steps = max(1, int(round(self.period_save / dt)))

        # Consider that we last saved at the current iteration
        # (init files / first save has already happened)
        self._it_last_save = it

    # Standard "every N steps" logic
    if it - self._it_last_save >= self._n_save_steps:
        self._it_last_save = it
        # Keep t_last_save consistent for end_of_simul and other code
        self.t_last_save = sim.time_stepping.t
        return True

    return False


