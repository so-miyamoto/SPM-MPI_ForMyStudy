import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize

d = np.loadtxt("dat/particles.dat")[::100,:]
NX = NY = 100

fig,axes = plt.subplots(3,1,layout="constrained")
axes[0].plot(d[:,0],d[:,2],"-")
axes[1].plot(d[:,0],d[:,4],"r:")
axes[2].plot(d[:,0],d[:,6],"b-.")
plt.xlabel(r"$t$")
plt.ylabel(r"$x,v,f$")
plt.savefig("v.png")
plt.close()


NX=NY=100

frames = 100

fig = plt.figure(figsize=(6,4),layout="constrained")
def func(frame):
  fig.clear()
  ax = fig.add_subplot(111)
  d = np.loadtxt(f"dat/fluid_{2000*(frame+1)}.dat",skiprows=0).reshape(NX,NY,5)
  # im0 = ax.pcolormesh(d[:,:,0],d[:,:,1],d[:,:,3],cmap="plasma",norm=Normalize(vmin=-0.8, vmax=0.2))
  # plt.pcolormesh(d[:,:,0],d[:,:,1],np.sqrt(d[:,:,2]**2+d[:,:,3]**2),cmap="plasma")
  # im0 = ax.contourf(d[:,:,0],d[:,:,1],d[:,:,3],cmap="plasma",norm=Normalize(vmin=-0.8, vmax=0.2))
  # im0 = ax.pcolormesh(d[:,:,0],d[:,:,1],np.sqrt(d[:,:,2]**2+d[:,:,3]**2),cmap="plasma",norm=Normalize(vmin=0.0, vmax=0.1))
  vmin,vmax = np.min(d[:,:,2]), np.max(d[:,:,2])
  im0 = ax.pcolormesh(d[:,:,0],d[:,:,1],d[:,:,2]*(1.0-d[:,:,4]),
    cmap="bwr",norm=Normalize(vmin=vmin, vmax=vmax))
  # im0 = ax.pcolormesh(d[:,:,0],d[:,:,1],d[:,:,4],cmap="plasma",norm=Normalize(vmin=0.0, vmax=1.0))
  cb0 = fig.colorbar(im0,ax=ax)
  # im0.set_clim(vmin=-0.8, vmax=0.2)
  # cb1.set_clim(0.0,1.0)
ani = FuncAnimation(fig, func, frames=range(0,frames,1), interval=100)
ani.save('uy.mp4')
plt.close()
