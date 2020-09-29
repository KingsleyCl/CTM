import UrbanStreet as us
from ControlVector import ControlVector_Webster
from CTM import CTM, Slice
import numpy as np
import matplotlib.pyplot as plt

Time_SignalPeriod = [900, 1800, 900]
TotalTimeStep = sum(Time_SignalPeriod)
LostTime = 5

Node, Link, Signal = us.config('urban data')
Slice(Link, Node, Signal, 1)

Control = ControlVector_Webster(len(Link), Signal, TotalTimeStep, Time_SignalPeriod, LostTime)
Inflow, Outflow, pho = CTM(Control, Link, Node, 1, TotalTimeStep)
np.savetxt('pho.csv', pho, delimiter=',', fmt='%f')

# pho = np.loadtxt('pho.csv', delimiter=',')
# X, Y = np.meshgrid(np.arange(300, 601), np.arange(1 / 120, 1 + 1 / 120, 1 / 120))
# plt.contourf(X, Y, pho[16:136, 300:601], cmap=plt.cm.Spectral, alpha=0.8)
X, Y = np.meshgrid(np.arange(pho.shape[1]), np.arange(pho.shape[0]))
plt.contourf(X, Y, pho, cmap=plt.cm.Spectral, alpha=0.8)
plt.show()
