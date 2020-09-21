import numpy as np


class node:
    def __init__(self, i, o, sp, si=None):
        self.InLink = tuple(i)
        self.OutLink = tuple(o)
        self.Split = np.array(sp).reshape((len(i), len(o)))
        self.Signal = si


class signal:
    def __init__(self, cyc, off, se, res):
        self.Cycle = cyc
        self.Offset = off
        self.StageEnd = np.array(se).reshape(1, 2)
        self.Restricted = np.array(res).reshape(1, 2)


class link:
    def __init__(self, fr, to, len, v, sf, kj, dem):
        self.FrNode = fr
        self.ToNode = to
        self.Length = len
        self.V = v
        self.SatFlow = sf
        self.kjam = kj
        self.Demand = dem
        self.kcrit = self.SatFlow / self.V
        self.W = self.SatFlow / (self.kjam - self.kcrit)


# def config(folder):
#     node_data = pd.read_csv(folder + '/node.csv')

#     def node_conduct(line):
#         # for sp in self.split:
#         #     if sum(sp) != 1:
#         #         print(self.in_link, self.out_link)
#         #         print(*self.split, sep='\n')
#         #         return
#         pass