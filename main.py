from UrbanStreet import config
from ControlVector import ControlVector_Webster
from CTM import CTM_matrix, Slice
import numpy as np
import matplotlib.pyplot as plt
from time import time

Time_SignalPeriod = [900, 1800, 900]
TotalTimeStep = sum(Time_SignalPeriod)
LostTime = 5

Node, Link, Signal = config('UrbanConfig_TCR.xlsx')
Slice(Link, Node, Signal, 1)

Control = ControlVector_Webster(len(Link), Signal, TotalTimeStep, Time_SignalPeriod, LostTime)

start = time()
Inflow, Outflow, pho = CTM_matrix(Control, Link, Node, 1, TotalTimeStep)
print(time() - start)

# figure
TimeMin, TimeMax = 1500, 1800  # range of time
LinkMin, LinkMax = 16, 136  # range of links
LocStep = 1 / (LinkMax - LinkMin)
X, Y = np.meshgrid(np.arange(TimeMin, TimeMax), np.arange(LocStep, 1 + LocStep, LocStep))
plt.contourf(X, Y, pho[LinkMin:LinkMax, TimeMin:TimeMax])
plt.colorbar()
plt.show()
