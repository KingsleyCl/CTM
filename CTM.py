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
import UrbanStreet as us


def CTM(control, Link, Node, dt, TotalTimeStep):
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

    for t in range(TotalTimeStep):
        # Calculate 'outflow' from each cell without restriction
        # (Step 1 in Kurzhanskiy et al., Eqn 2)
        for i in range(len(Link)):
            TotalSend[i, t] = min(Link[i].V * pho[i, t], Link[i].SatFlow * control[i, t])

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
                A = np.array(TotalSend[Node[n].InLink[0], t])
                for i in range(1, len(Node[n].InLink)):
                    A = np.hstack([A, TotalSend[Node[n].InLink[i], t]])
                TotalReceive[Node[n].OutLink[j], t] = A.dot(Node[n].Split[:, j].reshape(-1, 1))

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
                    else:
                        AdjustSplitSend[n][i, j, t] = 0

        # (Step 4 in Kurzhanskiy et al., Eqn 7)
        for n in range(len(Node)):
            for i in range(len(Node[n].InLink)):
                AdjustTotalSend[Node[n].InLink[i], t] = sum(AdjustSplitSend[n][i, :, t])

        for i in range(len(Link)):
            if Link[i].ToNode < 0:
                # recognized as a sink (i.e. no restraint downstream)
                AdjustTotalSend[i, t] = TotalSend[i, t]

        # Ensure node FIFO
        # (Step 5 in Kurzhanskiy et al., Eqn 8)
        for n in range(len(Node)):
            for i in range(len(Node[n].InLink)):
                TuneRatio[Node[n].InLink[i], t] = np.inf
                for j in range(len(Node[n].OutLink)):
                    if AdjustTotalSend[Node[n].InLink[i], t] * Node[n].Split[i, j] > 0:
                        r = AdjustSplitSend[n][i, j, t] / (AdjustTotalSend[Node[n].InLink(i), t] * Node[n].Split[i, j])
                    else:
                        r = 1
                    if r < TuneRatio[Node[n].InLink[i], t]:
                        TuneRatio[Node[n].InLink[i], t] = r

        # Calculate final outflow from each cell
        # (Step 5 in Kurzhanskiy et al., Eqn 8)
        for i in range(len(Link)):
            if Link[i].ToNode < 0:
                TuneRatio[i, t] = 1
            Outflow[i, t] = AdjustTotalSend[i, t] * TuneRatio[i, t]

        # Calculate inflow to each cell (including sources)
        # (Step 6 in Kurzhanskiy et al., Eqn 9)
        for n in range(len(Node)):
            for j in range(len(Node[n].OutLink)):
                A = np.array(TotalSend[Node[n].InLink[0], t])
                for i in range(1, len(Node[n].InLink)):
                    A = np.hstack([A, TotalSend[Node[n].InLink[i], t]])
                Inflow[Node[n].OutLink[j], t] = A.dot(Node[n].Split[:, j].reshape(-1, 1))

        for j in range(len(Link)):
            if Link[j].FrNode < 0:
                # demand profile resolution: 15-min
                # assume demand is normally distributed with variance = 10% of
                # mean
                if t // 900 >= len(Link[j].Demand):
                    Inflow[j, t] = Link[j].Demand[-1] + np.sqrt(0.1 * Link[j].Demand[-1]) * np.random.randn(1)
                else:
                    Inflow[j, t] = Link[j].Demand[t // 900] + np.sqrt(0.1 * Link[j].Demand[t // 900]) * np.random.randn(1)
                Inflow[j, t] = max(Inflow[j, t], 0)

        # Update density (Eqn 10 in Kurzhanski et al.)
        for i in range(len(Link)):
            pho[i, t + 1] = pho[i, t] + (Inflow[i, t] - Outflow[i, t]) * (dt / 3600) / Link[i].Length

        return Inflow, Outflow, pho


if __name__ == '__main__':
    Node = [us.node([0, 2], [1, 3], [0.7, 0.3] * 2, us.signal(120, 0, [60, 120], [5, 1]))]
    Link = [us.link(-1, 1, 0.01, 30, 1800, 230, [700, 850, 700, 0]),
            us.link(1, 2, 0.1, 30, 1800, 230, [0]),
            us.link(-1, 1, 0.01, 30, 1800, 230, [450, 300, 200, 0]),
            us.link(1, -1, 0.01, 30, 1800, 230, [0])]
    Inflow, Outflow, pho = CTM(np.array([1] * len(Link) * 3000).reshape(len(Link), -1), Link, Node, 1, 3000)
    np.savetxt('result.csv', np.around(Inflow), delimiter=',')
