import sys
import glob
import numpy as np
import netCDF4
import matplotlib.pyplot as plt


indir=sys.argv[1]


files=glob.glob(indir + "/state_phys*.nc")

it=0
for f in files :

### nature run
  nc = netCDF4.Dataset(f,'r')

#print(nc)
#print(nc.groups['state_phys'])
#for var in nc.variables[:]:
#    print(var)
#quit()
  vor = np.array(nc.groups['state_phys'].variables['rot'][:])
  nc.close

#vor=np.random.rand(256,256)

#print(vor)
#print(np.max(vor),np.min(vor))


#plt.contourf()

  x1=np.linspace(1,256,num=256)
  y1=np.linspace(1,256,num=256)
  x,y=np.meshgrid(x1,y1)

  cs=plt.pcolormesh(x,y,vor)
  #cs=plt.contourf(x.astype(np.float32),y.astype(np.float32),vor)
  cbar = plt.colorbar(cs,location='right')
  #cbar.set_label('m/s')
  fname="test_"+str(it).zfill(3)+".png"
  print(fname)
  plt.savefig(fname)
  plt.clf()
  it += 1

