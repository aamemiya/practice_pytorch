import sys
import numpy as np
import matplotlib.pyplot as plt
from fluidsim import load_sim_for_plot

indir=sys.argv[1]

sim = load_sim_for_plot(indir)

pf = sim.output.phys_fields
#print(pf)


sof = pf.set_of_phys_files       # SetOfPhysFieldFiles
sof.update_times()               # refresh file list
times = sof.times                # list of float times
#times = pf.get_times_saved()

E_series, Z_series = [], []
for t in times:
    ux,_ = pf.get_field_to_plot("ux", time=t)    # or compute from streamfunction if not saved
    uy,_ = pf.get_field_to_plot("uy", time=t)[:]
    rot,_ = pf.get_field_to_plot("rot", time=t)[:]   # vorticity
   # print(ux)
   # print(type(ux))
   # print(ux.shape)
   # quit()
    E_series.append(0.5 * np.mean(ux**2 + uy**2))
    Z_series.append(0.5 * np.mean(rot**2))

E_series = np.array(E_series)
Z_series = np.array(Z_series)

plt.plot(times, E_series, label="Kinetic energy")
plt.plot(times, Z_series, label="Enstrophy")
plt.xlabel("time"); plt.grid(True); plt.legend()
fname="monitor.png"
plt.savefig(fname)
plt.clf()
 

