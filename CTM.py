'''
Traffic model -- Cell transmission model
Reference: Kurzhanskiy A, Kwon J and Varaiya P (2009) Aurora Road Network
Modeler. In: Proceedings of 12th IFAC Symposium on Control in Transportation
Systems.

i - 'from' cell
j - 'to' link
Inflow(j,t)   - flow [veh/hr] into cell j during time interval t
Outflow(i,t)   - flow [veh/hr] out of cell i during time interval t
pho(i,t)    - density [veh/mile] in cell i by the end of time interval t

Initialize matrics
'''
import numpy as np
from copy import deepcopy
import UrbanStreet as us


def Slice(Link, Node, Signal, dt):
    '''A subroutine to slice links into smaller segments'''

    MaxNumCell = 60  # default maximum number of sub-cells

    OriginalNumLink = len(Link)
    for i in range(OriginalNumLink):
        FirstCellLink = len(Link)
        if Link[i].FrNode is not None and Link[i].ToNode is not None:
            # condition: length of sub-cell has to be greater than distance
            # travelled by free-flow speed times delta_t
            NumCell = min(MaxNumCell, int(Link[i].Length / (Link[i].V * dt / 3600)))

            Node.extend([us.node(InLink=[len(Link) + k], OutLink=[len(Link) + k + 1], Split=[1]) for k in range(1, NumCell)])

            for k in range(NumCell - 1, -1, -1):
                Link.append(deepcopy(Link[i]))
                Link[-1].FrNode, Link[-1].ToNode = len(Node) - k - 1, len(Node) - k
                Link[-1].Length /= NumCell
            Link[FirstCellLink].FrNode = Link[i].FrNode
            Link[-1].ToNode = Link[i].ToNode

            # Modify the Controller and node settings accordingly:
            LinkTmp = Node[Link[i].ToNode].InLink
            LinkTmp[np.where(LinkTmp == i)] = len(Link) - 1
            LinkTmp = Node[Link[i].FrNode].OutLink
            LinkTmp[np.where(LinkTmp == i)] = FirstCellLink
            for sig in Signal:
                sig.Restricted[np.where(sig.Restricted == i)] = len(Link) - 1


def CTM(Control, Link, Node, dt, TotalTimeStep):
    Inflow = np.zeros((len(Link), TotalTimeStep))
    Outflow = np.zeros((len(Link), TotalTimeStep))
    pho = np.zeros((len(Link), TotalTimeStep))

    TotalSend = np.zeros((len(Link), TotalTimeStep))
    TotalReceive = np.zeros((len(Link), TotalTimeStep))
    AdjustTotalSend = np.zeros((len(Link), TotalTimeStep))

    SplitSend = [np.zeros((len(Node[n].InLink), len(Node[n].OutLink), TotalTimeStep)) for n in range(len(Node))]
    AdjustSplitSend = [np.zeros((len(Node[n].InLink), len(Node[n].OutLink), TotalTimeStep)) for n in range(len(Node))]

    Available = np.zeros((len(Link), TotalTimeStep))

    TuneRatio = np.zeros((len(Link), TotalTimeStep))

    for t in range(TotalTimeStep - 1):
        # Calculate 'outflow' from each cell without restriction
        # (Step 1 in Kurzhanskiy et al., Eqn 2)
        for i in range(len(Link)):
            TotalSend[i, t] = min(Link[i].V * pho[i, t], Link[i].SatFlow * Control[i, t])

        # Calculate the splitted outflow from each cell without restriction
        # (Step 2 in Kurzhanskiy et al., Eqn 3)
        for n in range(len(Node)):
            for i in range(len(Node[n].InLink)):
                for j in range(len(Node[n].OutLink)):
                    SplitSend[n][i, j, t] = TotalSend[Node[n].InLink[i], t] * Node[n].Split[i, j]

        # Calculate flow reaching each cell without restriction
        # (Step 2 in Kurzhanskiy et al., Eqn 4)
        for n in range(len(Node)):
            for j in range(len(Node[n].OutLink)):
                A = np.array([TotalSend[Node[n].InLink[0], t]])
                for i in range(1, len(Node[n].InLink)):
                    A = np.hstack([A, TotalSend[Node[n].InLink[i], t]])
                TotalReceive[Node[n].OutLink[j], t] = A.dot(Node[n].Split[:, j])

        # Calculate 'available space' at downstream
        # (Step 3 in Kurzhanskiy et al., Eqn 5)
        for j in range(len(Link)):
            Available[j, t] = min(Link[j].SatFlow, Link[j].W * (Link[j].kjam - pho[j, t]))

        # Adjusted "Flow" after taking restriction into account
        # (Step 4 in Kurzhanskiy et al., Eqn 6)
        for n in range(len(Node)):
            for i in range(len(Node[n].InLink)):
                for j in range(len(Node[n].OutLink)):
                    if TotalReceive[Node[n].OutLink[j], t] > 0:
                        AdjustSplitSend[n][i, j, t] = min(TotalReceive[Node[n].OutLink[j], t], Available[Node[n].OutLink[j], t]) / TotalReceive[Node[n].OutLink[j], t] * SplitSend[n][i, j, t]

        # (Step 4 in Kurzhanskiy et al., Eqn 7)
        for n in range(len(Node)):
            for i in range(len(Node[n].InLink)):
                AdjustTotalSend[Node[n].InLink[i], t] = sum(AdjustSplitSend[n][i, :, t])

        for i in range(len(Link)):
            if Link[i].ToNode is None:
                # recognized as a sink (i.e. no restraint downstream)
                AdjustTotalSend[i, t] = TotalSend[i, t]

        # Ensure node FIFO
        # (Step 5 in Kurzhanskiy et al., Eqn 8)
        for n in range(len(Node)):
            for i in range(len(Node[n].InLink)):
                TuneRatio[Node[n].InLink[i], t] = 1
                for j in range(len(Node[n].OutLink)):
                    if AdjustTotalSend[Node[n].InLink[i], t] * Node[n].Split[i, j] > 0:
                        TuneRatio[Node[n].InLink[i], t] = min(TuneRatio[Node[n].InLink[i], t], AdjustSplitSend[n][i, j, t] / (AdjustTotalSend[Node[n].InLink[i], t] * Node[n].Split[i, j]))

        # Calculate final outflow from each cell
        # (Step 5 in Kurzhanskiy et al., Eqn 8)
        for i in range(len(Link)):
            if Link[i].ToNode is None:
                TuneRatio[i, t] = 1
            Outflow[i, t] = AdjustTotalSend[i, t] * TuneRatio[i, t]

        # Calculate inflow to each cell (including sources)
        # (Step 6 in Kurzhanskiy et al., Eqn 9)
        for n in range(len(Node)):
            for j in range(len(Node[n].OutLink)):
                A = np.array([Outflow[Node[n].InLink[0], t]])
                for i in range(1, len(Node[n].InLink)):
                    A = np.hstack([A, Outflow[Node[n].InLink[i], t]])
                Inflow[Node[n].OutLink[j], t] = A.dot(Node[n].Split[:, j])

        for j in range(len(Link)):
            if Link[j].FrNode is None:
                # demand profile resolution: 15-min
                # assume demand is normally distributed with variance = 10% of
                # mean
                k = min(t // 900, len(Link[j].Demand) - 1)
                Inflow[j, t] = max(Link[j].Demand[k] + np.sqrt(0.1 * Link[j].Demand[k]) * np.random.randn(1), 0)

        # Update density (Eqn 10 in Kurzhanski et al.)
        for i in range(len(Link)):
            pho[i, t + 1] = pho[i, t] + (Inflow[i, t] - Outflow[i, t]) * (dt / 3600) / Link[i].Length

    return Inflow, Outflow, pho


def CTM_matrix(Control, Link, Node, dt, TotalTimeStep):
    Inflow = np.zeros([len(Link), TotalTimeStep])
    Outflow = np.zeros([len(Link), TotalTimeStep])
    pho = np.zeros([len(Link), TotalTimeStep])

    Source = np.array([i for i, link in enumerate(Link) if link.FrNode is None])
    Destination = np.array([i for i, link in enumerate(Link) if link.ToNode is None])

    SourceDemandMatrix = np.vstack([Link[i].Demand for i in Source])

    VVector = np.array([link.V for link in Link])
    SatFlowVector = np.array([link.SatFlow for link in Link])
    WVector = np.array([link.W for link in Link])
    kjamVector = np.array([link.kjam for link in Link])
    LengthVector = np.array([link.Length for link in Link])

    SplitMatrix = np.zeros([len(Link)] * 2)
    for node in Node:
        SplitMatrix[[node.InLink.reshape(-1, 1), node.OutLink]] = node.Split

    for t in range(TotalTimeStep - 1):
        # Calculate the splitted outflow from each cell without restriction
        # (Step 1&2 in Kurzhanskiy et al., Eqn 2&3)
        DemandMatrix = SplitMatrix * np.minimum(SatFlowVector * Control[:, t], VVector * pho[:, t]).reshape(-1, 1)

        # Calculate flow reaching each cell without restriction
        # (Step 2 in Kurzhanskiy et al., Eqn 4)
        OutputDemand = np.sum(DemandMatrix, 0)

        # Calculate 'available space' at downstream
        # (Step 3 in Kurzhanskiy et al., Eqn 5)
        Available = np.minimum(SatFlowVector, WVector * (kjamVector - pho[:, t]))

        # Adjusted "Flow" after taking restriction into account
        # (Step 4 in Kurzhanskiy et al., Eqn 6)
        AdjustVector = np.minimum(Available, OutputDemand) / OutputDemand
        AdjustVector[np.where(OutputDemand == 0)] = 0
        AdjustDemandMatrix = DemandMatrix * AdjustVector
        AdjustInputDemand = np.sum(AdjustDemandMatrix, 1)

        # Calculate final outflow from each cell
        # (Step 4 in Kurzhanskiy et al., Eqn 7)
        AdjustMatrix = AdjustDemandMatrix / (SplitMatrix * AdjustInputDemand.reshape(-1, 1))
        AdjustMatrix[np.isnan(AdjustMatrix)] = 1
        # (Step 5 in Kurzhanskiy et al., Eqn 8)
        Outflow[:, t] = AdjustInputDemand * np.min(AdjustMatrix, 1)

        # Calculate inflow to each cell (except sources)
        # (Step 6 in Kurzhanskiy et al., Eqn 9)
        Inflow[:, t] = Outflow[:, t].dot(SplitMatrix)

        # Calculate source inflow to each cell
        k = min(t // 900, SourceDemandMatrix.shape[1] - 1)
        Inflow[Source, t] = np.maximum(SourceDemandMatrix[:, k] + np.sqrt(0.1 * SourceDemandMatrix[:, k]) * np.random.randn(SourceDemandMatrix.shape[0]), 0)

        # Calculate destination outflow to each cell
        Outflow[Destination, t] = np.minimum(SatFlowVector[Destination], VVector[Destination] * pho[Destination, t])

        # Update density (Eqn 10 in Kurzhanski et al.)
        pho[:, t + 1] = pho[:, t] + (Inflow[:, t] - Outflow[:, t]) * (dt / 3600) / LengthVector

    return Inflow, Outflow, pho
