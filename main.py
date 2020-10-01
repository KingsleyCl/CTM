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

# figure
TimeMin, TimeMax = 1500, 1800  # range of time
LinkMin, LinkMax = 16, 136  # range of links
LocStep = 1 / (LinkMax - LinkMin)
X, Y = np.meshgrid(np.arange(TimeMin, TimeMax), np.arange(LocStep, 1 + LocStep, LocStep))
plt.contourf(X, Y, pho[LinkMin:LinkMax, TimeMin:TimeMax])
plt.colorbar()
plt.show()
