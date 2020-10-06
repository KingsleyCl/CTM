import numpy as np


class node:
    def __init__(self, InLink, OutLink, Split, Signal=None):
        self.InLink = np.array(InLink, dtype=int) - 1
        self.OutLink = np.array(OutLink, dtype=int) - 1
        self.Split = np.array(Split, dtype=float).reshape((len(InLink), len(OutLink)))


class link:
    def __init__(self, FrNode, ToNode, Length, V, SatFlow, kjam, Demand):
        if FrNode == '-1':
            self.FrNode = None
        else:
            self.FrNode = int(FrNode) - 1
        if ToNode == '-1':
            self.ToNode = None
        else:
            self.ToNode = int(ToNode) - 1
        self.Length = float(Length)
        self.V = float(V)
        self.SatFlow = float(SatFlow)
        self.kjam = float(kjam)
        self.Demand = np.array(Demand, dtype=int)
        self.kcrit = self.SatFlow / self.V
        self.W = self.SatFlow / (self.kjam - self.kcrit)


class signal:
    def __init__(self, Node, Restricted, Condition, Offset, GreenSplit):
        self.Node = int(Node) - 1
        self.Restricted = np.array(Restricted, dtype=int) - 1
        self.Condition = np.array(Condition, dtype=int)
        self.Offset = int(Offset)
        self.GreenSplit = np.array(GreenSplit, dtype=int).reshape(-1, len(Restricted))


def config(folder):
    # read Node
    Data = np.loadtxt(folder + '/node.csv', dtype=str, delimiter=',', skiprows=1)
    Node = [node(line[1].split(';'), line[2].split(';'), line[3].split(';')) for line in Data]

    # read Link
    Data = np.loadtxt(folder + '/link.csv', dtype=str, delimiter=',', skiprows=1)
    Link = [link(*line[1:7], line[7].split(';')) for line in Data]

    # read signal
    Data = np.loadtxt(folder + '/signal.csv', dtype=str, delimiter=',', skiprows=1)
    Signal = [signal(line[1], line[2].split(';'), line[3].split(';'), line[4], line[5].split(';')) for line in Data]

    return Node, Link, Signal
