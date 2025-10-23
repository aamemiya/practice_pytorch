
import math                              # 数学ライブラリ
import numpy as np                       # 数値計算用の配列
import numpy.linalg as LA                # 行列計算ライブラリ
import netCDF4 


class Lorenz96:
  """
  This class provides the Lorenz96 model equations and the 4th-order Runge-Kutta solver.
  """
  def __init__(self, n = 40, f = 8, dt = 0.005, init_x = None , amp_const = 0 ):
    self.n = n
    self.f = f
    self.dt = dt
    self.amp = amp_const
    if init_x is not None:
      self.x = init_x
    else:
      self.x = np.zeros(n, dtype=np.float64)
    # temporary memory
    self.tmpx = np.zeros(n + 3, dtype=np.float64)
    self.tmpdx = np.zeros(n, dtype=np.float64)
    return
  
  def dx_dt(self, y):
    n = y.shape[0]
    m = n + 3
    self.tmpx[2:(m - 1)] = y[:]
    # cyclic boundary condition
    self.tmpx[0:2] = y[(n - 2):n]
    self.tmpx[m - 1] = y[0]
    #self.tmpdx = (self.tmpx[3:m] - self.tmpx[0:(m - 3)]) * self.tmpx[1:(m - 2)] - self.tmpx[2:(m - 1)] + self.f
    self.tmpdx[:] = self.tmpx[3:m]
    self.tmpdx   -= self.tmpx[0:(m - 3)]
    self.tmpdx   *= self.tmpx[1:(m - 2)]
    self.tmpdx   -= self.tmpx[2:(m - 1)]
    self.tmpdx   += self.f
    self.tmpdx   += self.amp * np.sin(2*np.pi*np.arange(n)/n)
    return self.tmpdx

  def runge_kutta(self):
    dx1 = self.dx_dt(self.x)
    x1  = self.x + dx1 * (self.dt * 0.5)
    dx2 = self.dx_dt(x1)
    x2  = self.x + dx2 * (self.dt * 0.5)
    dx3 = self.dx_dt(x2)
    x3  = self.x + dx3 * self.dt
    dx4 = self.dx_dt(x3)
    self.x += (dx1 + 2.0 * (dx2 + dx3) + dx4) * (self.dt / 6.0)
    return self.x

  def save_snap(self,ncname):
    nc = netCDF4.Dataset(ncname,'w',format='NETCDF3_CLASSIC')
    nc.createDimension('x',self.n)
    x_in = nc.createVariable('x',np.dtype('float64').char,('x'))
    v_in = nc.createVariable('v',np.dtype('float64').char,('x'))
    x_in[:] = np.array(range(1,1+self.n))
    v_in[:] = self.x
    nc.close 

  def load_snap(self,ncname):
    nc = netCDF4.Dataset(ncname,'r',format='NETCDF4')
    self.x = np.array(nc.variables['v'][:], dtype=type(np.float64))
    nc.close 

class Lorenz96_add:
  """
  This class provides the Lorenz96 model equations and the 4th-order Runge-Kutta solver.
  """
  def __init__(self, n = 40, f = 8, dt = 0.005, init_x = None , amp_add = 0 , amp_add_2 = None , mode = None ):
    self.n = n
    self.f = f
    self.dt = dt
    self.amp = amp_add
    self.amp2 = f
    self.t = 0
    if amp_add_2 is not None:
      self.amp2 = amp_add_2
    self.mode = mode
    if init_x is not None:
      self.x = init_x
    else:
      self.x = np.zeros(n, dtype=np.float64)
    # temporary memory
    self.tmpx = np.zeros(n + 3, dtype=np.float64)
    self.tmpdx = np.zeros(n, dtype=np.float64)
    return
  
  def dx_dt(self, y):
    n = y.shape[0]
    m = n + 3
    self.tmpx[2:(m - 1)] = y[:]
    # cyclic boundary condition
    self.tmpx[0:2] = y[(n - 2):n]
    self.tmpx[m - 1] = y[0]
    #self.tmpdx = (self.tmpx[3:m] - self.tmpx[0:(m - 3)]) * self.tmpx[1:(m - 2)] - self.tmpx[2:(m - 1)] + self.f
    self.tmpdx[:] = self.tmpx[3:m]
    self.tmpdx   -= self.tmpx[0:(m - 3)]
    self.tmpdx   *= self.tmpx[1:(m - 2)]
    self.tmpdx   -= self.tmpx[2:(m - 1)]
    self.tmpdx   += self.f
    if self.mode == "const":
      self.tmpdx   += self.amp 
    if self.mode == "sin":
      self.tmpdx   += self.amp * np.sin(2*np.pi*np.arange(n)/n)
    if self.mode == "sint":
      self.tmpdx   += self.amp * np.sin(2*np.pi*self.t/self.amp2)
    if self.mode == "linear":
      self.tmpdx   += self.amp * self.tmpx[2:(m - 1)]
    if self.mode == "third_order":
      self.tmpdx   += self.amp * (self.tmpx[2:(m - 1)] - 1/(self.amp2)**2 * self.tmpx[2:(m - 1)]**3)
    if self.mode == "step":
      self.tmpdx   += self.amp * np.heaviside( self.tmpx[2:(m - 1)] - self.amp2, 0 )
    if self.mode == "advection":
      self.tmpdx   += self.amp * (self.tmpx[3:m] - self.tmpx[0:(m - 3)]) * self.tmpx[1:(m - 2)]
    return self.tmpdx

  def runge_kutta(self):
    dx1 = self.dx_dt(self.x)
    x1  = self.x + dx1 * (self.dt * 0.5)
    dx2 = self.dx_dt(x1)
    x2  = self.x + dx2 * (self.dt * 0.5)
    dx3 = self.dx_dt(x2)
    x3  = self.x + dx3 * self.dt
    dx4 = self.dx_dt(x3)
    self.x += (dx1 + 2.0 * (dx2 + dx3) + dx4) * (self.dt / 6.0)
    self.t += self.dt 
    return self.x

  def save_snap(self,ncname):
    nc = netCDF4.Dataset(ncname,'w',format='NETCDF3_CLASSIC')
    nc.createDimension('x',self.n)
    x_in = nc.createVariable('x',np.dtype('float64').char,('x'))
    v_in = nc.createVariable('v',np.dtype('float64').char,('x'))
    x_in[:] = np.array(range(1,1+self.n))
    v_in[:] = self.x
    nc.close 

  def load_snap(self,ncname):
    nc = netCDF4.Dataset(ncname,'r',format='NETCDF4')
    self.x = np.array(nc.variables['v'][:], dtype=type(np.float64))
    nc.close 


