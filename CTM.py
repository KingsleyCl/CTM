# Traffic model -- Cell transmission model
# Reference: Kurzhanskiy A, Kwon J and Varaiya P (2009) Aurora Road Network
# Modeler. In: Proceedings of 12th IFAC Symposium on Control in Transportation
# Systems.


# i - 'from' cell 
# j - 'to' link
# Inflow(j,t)   - flow [veh/hr] into cell j during time interval t 
# Outflow(i,t)   - flow [veh/hr] out of cell i during time interval t  
# pho(i,t)    - density [veh/mile] in cell i by the end of time interval t 

# Initialize matrics

import numpy as np

Inflow = np.zeros((length(Link), TotalTimeStep), np.int)
Outflow = np.zeros((length(Link), TotalTimeStep), np.int)
pho = np.zeros((length(Link), TotalTimeStep))

TotalSend = np.zeros((length(Link), TotalTimeStep), np.int)
TotalReceive = np.zeros((length(Link), TotalTimeStep), np.int)

Available = np.zeros((len(Link), TotalTimeStep))

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
                Node[n].SplitSend[i, j, t] = TotalSend[Node[n].InLink[i], t] * Node[n].Split[i, j]

    # Calculate flow reaching each cell without restriction 
    # (Step 2 in Kurzhanskiy et al., Eqn 4)
    for n in range(len(Node)):
        for j in range(len(Node[n].OutLink)):
            A = []
            for i in range(len(Node[n].InLink)):
                A += TotalSend[Node[n].InLink[i], t]
            TotalReceive[Node[n].OutLink[j], t] = A * Node[n].Split[:, j]
    
    # Calculate 'available space' at downstream
    # (Step 3 in Kurzhanskiy et al., Eqn 5)
    for j in range(len(Link)):
        Available[j, t] = min(Link[j].SatFlow, Link[j].W * (Link[j].kjam - pho[j,t]))

    # Adjusted "Flow" after taking restriction into account 
    # (Step 4 in Kurzhanskiy et al., Eqn 6)
    AdjustSplitSend = []
    for n in range(len(Node)):
        AdjustSplitSend.append(np.zeros((len(Node[n].InLink), len(Node[n].OutLink, TotalTimeStep))))
        for i in range(len(Node[n].InLink)):
            for j in range(len(Node[n].OutLink)):
                if TotalReceive[Node[n].OutLink[j], t] > 0:
                    AdjustNode[n].SplitSend[i, j, t] = min(TotalReceive[Node[n].OutLink[j], t], Available[Node[n].OutLink[j], t]) / TotalReceive[Node[n].OutLink[j], t] * Node[n].SplitSend[i, j, t]
                else:
                    AdjustNode[n].SplitSend[i, j, t] = 0
        
    # (Step 4 in Kurzhanskiy et al., Eqn 7)
    for n in range(len(Node)):
        for i in range(len(Node[n].InLink)):
            AdjustTotalSend(Node[n].InLink(i),t) = sum(AdjustNode[n].SplitSend[i, :, t])
    
    for i in range(len(Link)):
        if Link(i).ToNode < 0:
            # recognized as a sink (i.e. no restraint downstream)
            AdjustTotalSend[i, t] = TotalSend[i, t]
    
    # Ensure node FIFO 
    # (Step 5 in Kurzhanskiy et al., Eqn 8)
    for n in range(len(Node)):
        for i in range(len(Node[n].InLink)):
            TuneRatio(Node[n].InLink[i], t) = np.inf
            for j in range(len(Node[n].OutLink)):
                if AdjustTotalSend(Node[n].InLink[i], t) * Node[n].Split(i, j) > 0:
                    r = AdjustNode[n].SplitSend[i,j,t] / (AdjustTotalSend[Node[n].InLink(i), t] * Node[n].Split[i, j])
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
            A = []
            for i in range(len(Node[n].InLink)):
                A += Outflow[Node[n].InLink[i], t]
            Inflow[Node[n].OutLink[j], t] = A * Node[n].Split[:, j]
    
    for j in range(len(Link)):
        if Link[j].FrNode < 0:
            # demand profile resolution: 15-min
            # assume demand is normally distributed with variance = 10% of
            # mean 
            if t / 900 + 1 > len(Link[j].Demand):
                Inflow[j, t] = Link[j].Demand[end] + np.sqrt(0.1 * Link[j].Demand[end]) * np.random.randn(1)
            else:
                Inflow[j, t] = Link[j].Demand(np.floor(t / 900) + 1) + np.sqrt(0.1 * Link[j].Demand(floor(t / 900) + 1)) * np.random.randn(1)
            Inflow[j, t] = max(Inflow[j, t], 0
    
    # Update density (Eqn 10 in Kurzhanski et al.)
    for i in range(len(Link)):
        pho[i, t + 1] = pho[i, t] + (Inflow[i, t] - Outflow[i, t]) * (dt / 3600) / Link[i].Length
