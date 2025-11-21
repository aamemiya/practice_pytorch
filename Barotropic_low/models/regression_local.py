import os, sys, glob
import netCDF4 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy.linalg as LA             

# model size
nx  = 128
nx_low  = 32

scale=round(nx/nx_low)

ncdir="/data/Barotropic/output/nature/control/"
ncdir_low="/data/Barotropic/output/forecast_x4/"

nsmp_train=10
nsmp_test=10

width=3

vor_colors = [
    "#000066", 
    "#0000cc", 
    "#0000ff", 
    "#0066ff",  
    "#3399ff",  
    "#66ccff",  

    "#ffffff", 
    "#ffffff", 

    "#ffff99", 
    "#ffcc66", 
    "#ff9933", 
    "#ff3300",  
    "#ff0000",  
    "#800000",  
]
vor_colormap = matplotlib.colors.ListedColormap(vor_colors[1:-1])
vor_colormap.set_over(vor_colors[-1])
vor_colormap.set_under(vor_colors[0])

vor_levels = np.arange(-12,14,2)
vor_norm=matplotlib.colors.BoundaryNorm(vor_levels, len(vor_levels))

u_levels = np.arange(-12,14,2)*0.5
u_norm=matplotlib.colors.BoundaryNorm(u_levels, len(u_levels))




def make_ds(arrayin,scale):
    arrayout=np.zeros( arrayin.shape[:-2] + (arrayin.shape[-2]*scale,arrayin.shape[-1]*scale))
    for i in range(scale): 
        for j in range(scale):
            arrayout[...,i::scale,j::scale]=arrayin[...,:,:]
    return arrayout


def slice_cyclic(A, row_start, row_stop, col_start, col_stop):
    row_idx = np.arange(row_start, row_stop)
    col_idx = np.arange(col_start, col_stop)
    return np.take(np.take(A, row_idx, axis=-2, mode="wrap"),
                   col_idx, axis=-1, mode="wrap")

def vor2d(u,v,dx,dy):
    dvdx = np.gradient(v, dx, axis=-1)
    dudy = np.gradient(u, dy, axis=-2)
    return dvdx - dudy

files=glob.glob(os.path.join(ncdir + "/state_phys*.nc"))
tstamp_smp=files[0].split("/")[-1].replace("state_phys_t","").split(".")[0:2]
digit_all=str(len(tstamp_smp[0])+len(tstamp_smp[1])+1)
digit_dec=len(tstamp_smp[1])

dirs_fcst_low=glob.glob(os.path.join(ncdir_low,"[0-9]*.[0-9]*"))

print("demo sample from ",dirs_fcst_low[nsmp_train])

usmp=[]
vsmp=[]
ulsmp=[]
vlsmp=[]
utest=[]
vtest=[]
ultest=[]
vltest=[]


## Train
for ismp, dir_fcst_low in enumerate(dirs_fcst_low[:nsmp_train]) :
    files_low=glob.glob(os.path.join(dir_fcst_low + "/state_phys*.nc"))
    usmp.append([])
    vsmp.append([])
    ulsmp.append([])
    vlsmp.append([])
    for fl in files_low : 
    
# load nature 
        tstamp=fl.split("/")[-1].replace("state_phys_t","").replace(".nc","")

        if tstamp[-1] == "4" : break ### TORI AEZU
        ftstamp=float(tstamp)
        tstamp_n=format(f"{ftstamp:0{digit_all}.{digit_dec}f}")

        fn = os.path.join(ncdir,fl.split("/")[-1]).replace(tstamp,tstamp_n)
        nc  = netCDF4.Dataset(fn,'r',format='NETCDF4')
        ncl = netCDF4.Dataset(fl,'r',format='NETCDF4')
        ulsmp[ismp].append(np.array(ncl["state_phys"].variables['ux'][:]))
        vlsmp[ismp].append(np.array(ncl["state_phys"].variables['uy'][:]))
        usmp[ismp].append(np.array(nc["state_phys"].variables['ux'][:]))
        vsmp[ismp].append(np.array(nc["state_phys"].variables['uy'][:]))
        nc.close 
        ncl.close 

## Ttest
for ismp, dir_fcst_low in enumerate(dirs_fcst_low[nsmp_train:nsmp_train+nsmp_test]) :
    files_low=glob.glob(os.path.join(dir_fcst_low + "/state_phys*.nc"))
    utest.append([])
    vtest.append([])
    ultest.append([])
    vltest.append([])
    for fl in files_low : 
    
# load nature 
        tstamp=fl.split("/")[-1].replace("state_phys_t","").replace(".nc","")

        if tstamp[-1] == "4" : break ### TORI AEZU
        ftstamp=float(tstamp)
        tstamp_n=format(f"{ftstamp:0{digit_all}.{digit_dec}f}")

        fn = os.path.join(ncdir,fl.split("/")[-1]).replace(tstamp,tstamp_n)
        nc  = netCDF4.Dataset(fn,'r',format='NETCDF4')
        ncl = netCDF4.Dataset(fl,'r',format='NETCDF4')
        ultest[ismp].append(np.array(ncl["state_phys"].variables['ux'][:]))
        vltest[ismp].append(np.array(ncl["state_phys"].variables['uy'][:]))
        utest[ismp].append(np.array(nc["state_phys"].variables['ux'][:]))
        vtest[ismp].append(np.array(nc["state_phys"].variables['uy'][:]))
        nc.close 
        ncl.close 

uvsmp=np.stack((np.array(usmp),np.array(vsmp)),axis=-3)
uvlsmp=np.stack((np.array(ulsmp),np.array(vlsmp)),axis=-3)
uvtest=np.stack((np.array(utest),np.array(vtest)),axis=-3)
uvltest=np.stack((np.array(ultest),np.array(vltest)),axis=-3)

uvlsmp_ds=make_ds(uvlsmp,scale)
uvltest_ds=make_ds(uvltest,scale)

nt=uvsmp.shape[1]

for it in [0, 2, 4, 6]: 
    smp_knl=[]
    smp_knl_predictor=[]
    test_knl=[]
    test_knl_predictor=[]
    for j in range(nx_low):
        for i in range(nx_low): 
            smp_knl.append(  uvsmp[:,it,:,scale*j:scale*(j+1),scale*i:scale*(i+1)] - uvlsmp_ds[:,it,:,scale*j:scale*(j+1),scale*i:scale*(i+1)]  ) 
            smp_knl_predictor.append( slice_cyclic(uvlsmp[:,it,:,:,:],j-width+1,j+width,i-width+1,i+width) ) 
            test_knl.append(  uvtest[:,it,:,scale*j:scale*(j+1),scale*i:scale*(i+1)] - uvltest_ds[:,it,:,scale*j:scale*(j+1),scale*i:scale*(i+1)]  ) 
            test_knl_predictor.append( slice_cyclic(uvltest[:,it,:,:,:],j-width+1,j+width,i-width+1,i+width) ) 

    smp_knl=np.array(smp_knl).reshape(nx_low*nx_low*nsmp_train,2*scale*scale)
    smp_knl_predictor=np.array(smp_knl_predictor).reshape(nx_low*nx_low*nsmp_train,2*(2*width-1)**2)
    test_knl=np.array(test_knl).reshape(nx_low*nx_low*nsmp_test,2*scale*scale)
    test_knl_predictor=np.array(test_knl_predictor).reshape(nx_low*nx_low*nsmp_test,2*(2*width-1)**2)
  
    data_input_ext=np.concatenate((smp_knl_predictor,np.ones((smp_knl_predictor.shape[0],1))),axis=1)
    bmatinv=LA.inv(data_input_ext.transpose() @ data_input_ext)
    wmat = smp_knl.transpose() @ ( data_input_ext @ bmatinv )

#  plt.scatter(smp_knl_predictor[:,7],smp_knl[:,1,1])
#  plt.savefig("scat.png")
#  plt.clf()

 
    data_input_ext=np.concatenate((test_knl_predictor,np.ones((test_knl_predictor.shape[0],1))),axis=1)
 
    data_predict=data_input_ext @ wmat[:,:].transpose() 
    test_knl_res=test_knl-data_predict
  
    test_mean=np.mean(test_knl,axis=0)
    test_std=np.std(test_knl,axis=0)
    res_mean=np.mean(test_knl_res,axis=0)
    res_std=np.std(test_knl_res,axis=0)

    var_ratio = 100*(1.0-(np.mean(res_std**2) / np.mean(test_std**2)))

    print("it = ",it)
    print("mean test,residual",test_mean,res_mean)
    print("std test, residual",test_std,res_std)
    print("explained variance : ",var_ratio)

    ### sample output 
    utest_rep=np.zeros((nx,nx))
    vtest_rep=np.zeros((nx,nx))
    for j in range(nx_low): 
        for i in range(nx_low): 
            utest_rep[scale*j:scale*(j+1),scale*i:scale*(i+1)]  =  uvltest_ds[0,it,0,scale*j:scale*(j+1),scale*i:scale*(i+1)]  + data_predict[(nx_low*j+i)*nsmp_test+0,:scale*scale].reshape((scale,scale))
            vtest_rep[scale*j:scale*(j+1),scale*i:scale*(i+1)]  =  uvltest_ds[0,it,1,scale*j:scale*(j+1),scale*i:scale*(i+1)]  + data_predict[(nx_low*j+i)*nsmp_test+0,scale*scale:].reshape((scale,scale))


    utest_high=uvtest[0,it,0,:,:]   
    vtest_high=uvtest[0,it,1,:,:]   
    utest_low=uvltest[0,it,0,:,:]   
    vtest_low=uvltest[0,it,1,:,:]   

#    utest_high=uvtest[0,it,0,:,:] - uvltest_ds[0,it,0,:,:]  
#    utest_rep=utest_rep[:,:] - uvltest_ds[0,it,0,:,:]  

    dx=2.0*np.pi/nx
    vor_rep=vor2d(utest_rep,vtest_rep,dx,dx)
    vor_high=vor2d(utest_high,vtest_high,dx,dx)
    vor_low=vor2d(utest_low,vtest_low,scale*dx,scale*dx)
    print(np.max(vor_rep[2:-2,2:-2]),np.min(vor_rep[2:-2,2:-2]))
    print(np.max(vor_high[2:-2,2:-2]),np.min(vor_high[2:-2,2:-2]))
    print(np.max(vor_low[2:-2,2:-2]),np.min(vor_low[2:-2,2:-2]))
    print(np.max(utest_rep),np.min(utest_rep))
    print(np.max(utest_high),np.min(utest_high))
    print(np.max(utest_low),np.min(utest_low))

    x1=np.linspace(1,nx,num=nx)
    y1=np.linspace(1,nx,num=nx)
    x,y=np.meshgrid(x1,y1)

    cs=plt.pcolormesh(x,y,vor_rep, norm=vor_norm,cmap=vor_colormap)
    cbar = plt.colorbar(cs,location='right')
    fname="vor_rep.png"
    print(fname)
    plt.savefig(fname)
    plt.clf()
    cs=plt.pcolormesh(x,y,vor_high, norm=vor_norm,cmap=vor_colormap)
    cbar = plt.colorbar(cs,location='right')
    fname="vor_high.png"
    print(fname)
    plt.savefig(fname)
    plt.clf()

    x1=np.linspace(1,nx_low,num=nx_low)
    y1=np.linspace(1,nx_low,num=nx_low)
    xl,yl=np.meshgrid(x1,y1)

    cs=plt.pcolormesh(xl,yl,vor_low, norm=vor_norm,cmap=vor_colormap)
    cbar = plt.colorbar(cs,location='right')
    fname="vor_low.png"
    print(fname)
    plt.savefig(fname)
    plt.clf()

    cs=plt.pcolormesh(x,y,utest_rep, norm=u_norm,cmap=vor_colormap)
    cbar = plt.colorbar(cs,location='right')
    fname="u_rep.png"
    print(fname)
    plt.savefig(fname)
    plt.clf()
    cs=plt.pcolormesh(x,y,utest_high, norm=u_norm,cmap=vor_colormap)
    cbar = plt.colorbar(cs,location='right')
    fname="u_high.png"
    print(fname)
    plt.savefig(fname)
    plt.clf()
    cs=plt.pcolormesh(xl,yl,utest_low, norm=u_norm,cmap=vor_colormap)
    cbar = plt.colorbar(cs,location='right')
    fname="u_low.png"
    print(fname)
    plt.savefig(fname)
    plt.clf()


    quit()

quit()




