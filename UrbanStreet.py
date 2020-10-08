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


def config(file):
    import pandas as pd
    # read Node
    data = pd.read_excel(file, sheet_name=0, index_col=0, header=0, skiprows=0, dtype=str)
    Node = []
    data.apply(lambda x: Node.append(node(x[0].split(';'), x[1].split(';'), x[2].split(';'))), 1)

    # read Link
    data = pd.read_excel(file, sheet_name=1, index_col=0, header=0, skiprows=0, dtype=str)
    Link = []
    data.apply(lambda x: Link.append(link(*x[:6], x[6].split(';'))), 1)

    # read signal
    data = pd.read_excel(file, sheet_name=2, index_col=0, header=0, skiprows=0, dtype=str)
    Signal = []
    data.apply(lambda x: Signal.append(signal(x[0], x[1].split(';'), x[2].split(';'), x[3], x[4].split(';'))), 1)

    return Node, Link, Signal
