from fluidsim.solvers.ns2d.solver import Simul
import numpy as np, math
from netCDF4 import Dataset
from types import MethodType
from fluidsim.base.output.base import SpecificOutput

# Keep a reference to the original method
_orig_has_to_online_save = SpecificOutput._has_to_online_save

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

# Install the patch globally
SpecificOutput._has_to_online_save = _has_to_online_save_every_n_steps


params = Simul.create_default_params()
params.oper.nx = params.oper.ny = 128
params.oper.Lx = params.oper.Ly = 2.0 * 3.141592653589793
params.time_stepping.t_end = 100.0
params.time_stepping.deltat0 = 0.004
params.time_stepping.USE_CFL = False

digits=abs(math.floor(np.log10(params.time_stepping.deltat0)))

params.forcing.enable = True
params.forcing.type = "tcrandom"      # e.g. time-correlated random ring forcing
params.forcing.nkmin_forcing = 4
params.forcing.nkmax_forcing = 5
params.forcing.normalized.constant_rate_of = "energy"
params.forcing.forcing_rate = 0.2        # set target energy input rate
params.nu_4 = 4.0e-6                   # hyperviscosity (example)
params.nu_m4 = 5e-2                   # hypoviscosity (example)

params.output.sub_directory="output"
params.output.periods_save.phys_fields = 0.4

n_save = int(round( params.output.periods_save.phys_fields/ params.time_stepping.deltat0))

params.init_fields.type="noise"
params.init_fields.noise.length=0.0
sim = Simul(params)
op = sim.oper

#pf = sim.output.phys_fields
#pf._n_save = n_save
#pf._it_last_save = sim.time_stepping.it
#
#def _has_to_online_save_it(self):
#    it = self.sim.time_stepping.it
#    # First call: _it_last_save == 0, so first save after n_save steps
#    if it >= self._it_last_save + self._n_save:
#        self._it_last_save = it
#        return True
#    return False
#
#pf._has_to_online_save = MethodType(_has_to_online_save_it, pf)

sim.time_stepping.start()
