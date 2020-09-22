import UrbanStreet as us
from CTM import CTM
import numpy as np

# , us.signal(120, 0, [60, 120], [5, 1]), us.signal(120, 0, [60, 120], [7, 2])
Node = [us.node([0], [1], [1]),
        us.node([1], [2], [1])]
Link = [us.link(-1, 0, 0.01, 30, 1800, 230, [700, 850, 700, 0]),
        us.link(0, 1, 0.1, 30, 1800, 230, [0]),
        us.link(1, -1, 0.01, 30, 1800, 230, [450, 300, 200, 0]),]
        # us.link(1, -1, 0.01, 30, 1800, 230, [0])
Inflow, Outflow, pho = CTM(np.array([1] * len(Link) * 3000).reshape(len(Link), -1), Link, Node, 1, 3000)
np.savetxt('result.csv', np.around(pho), delimiter=',')
print(pho)
