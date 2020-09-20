import pandas as pd
import numpy as np


class node:
    def __init__(self, i, o, sp, si=None):
        self.in_link = tuple(i)
        self.out_link = tuple(o)
        self.split = tuple(tuple(sp[k * len(o):(k + 1) * len(o)]) for k in range(len(i)))
        self.signal = si
        self.SplitSend = np.zeros((len(self.InLink), len(self.OutLink, TotalTimeStep)))


class signal:
    def __init__(self, node, cycle, off, se, res):
        self.cycle = cycle
        self.offset = off
        self.stage_end = se
        self.restricted = res


class link:
    def __init__(self, fr, to, len, v, sf, kj, dem):
        self.fr_node = fr
        self.to_node = to
        self.length = len
        self.v = v
        self.sat_flow = sf
        self.kjam = kj
        self.demand = dem


def config(folder):
    node_data = pd.read_csv(folder + '/node.csv')

    def node_conduct(line):
        # for sp in self.split:
        #     if sum(sp) != 1:
        #         print(self.in_link, self.out_link)
        #         print(*self.split, sep='\n')
        #         return
        pass