import numpy as np
import pandas as pd


def config(file):
    def minus1(obj):
        return obj - 1

    def split2int(value):
        if type(value) == str:
            return np.array(value.split(';'), dtype=int)
        else:
            return np.array([value], dtype=int)

    def split2float(value):
        if type(value) == str:
            return np.array(value.split(';'), dtype=float)
        else:
            return np.array([value], dtype=float)

    Node = pd.read_excel(file, sheet_name=0, index_col=0, header=0, skiprows=0)
    Node = Node.reset_index(drop=True)
    Node[['InLink', 'OutLink']] = Node[['InLink', 'OutLink']].applymap(split2int)
    Node['Split'] = Node['Split'].map(split2float)

    def reshape1(node):
        node['Split'] = node['Split'].reshape(node['InLink'].shape[0], node['OutLink'].shape[0])
        return node

    Node = Node.apply(reshape1, 1)
    Node.loc[:, ['InLink', 'OutLink']] = Node[['InLink', 'OutLink']].applymap(minus1)

    Link = pd.read_excel(file, sheet_name=1, index_col=0, header=0, skiprows=0)
    Link = Link.reset_index(drop=True)
    Link['Demand'] = Link['Demand'].map(split2int)
    Link.loc[:, ['FrNode', 'ToNode']] = Link[['FrNode', 'ToNode']].applymap(minus1)
    Link['kcrit'] = Link['SatFlow'] / Link['V']
    Link['W'] = Link['SatFlow'] / (Link['kjam'] - Link['kcrit'])

    Signal = pd.read_excel(file, sheet_name=2, index_col=0, header=0, skiprows=0)
    Signal = Signal.reset_index(drop=True)
    Signal.loc[:, ['Restricted', 'Condition', 'GreenSplit']] = Signal[['Restricted', 'Condition', 'GreenSplit']].applymap(split2int)

    def reshape2(signal):
        signal['GreenSplit'] = signal['GreenSplit'].reshape(-1, Signal.loc[1, 'Restricted'].shape[0])
        return signal

    Signal = Signal.apply(reshape2, 1)
    Signal.loc[:, ['Node', 'Restricted']] = Signal[['Node', 'Restricted']].applymap(minus1)

    return Node, Link, Signal


def ControlVector_Webster(LenLink, Signal, TotalTimeStep, Time_SignalPeriod, LostTime):
    '''
    Control vector
    binary: 0 - RED 1-GREEN
    each link should only be associated with one Control vector
    Generic coding of Webster plan with offsets
    '''
    def offset(signal):
        ControlVector = np.array([]).reshape(-1, 2)  # Initialize the Control vector of each signal

        for j in range(signal['GreenSplit'].shape[0]):
            CycleTime = sum(signal['GreenSplit'][j, :]) + 2 * LostTime  # Calculate Cycle time

            Cycle = np.zeros((CycleTime, 2))  # Initialize signal
            Cycle[:signal['GreenSplit'][j, 0], 0] = 1  # First stage with green
            Cycle[signal['GreenSplit'][j, 0] + LostTime:signal['GreenSplit'][j, 0] + LostTime + signal['GreenSplit'][j, 1], 1] = 1  # Second stage with green

            # Add the thread in the head of signal as offsets
            OffSetSignal = Cycle[-signal['Offset'] % CycleTime:]

            CycleNum = Time_SignalPeriod[j] // CycleTime + 1  # Calculate the number of cycles

            CycleList = np.vstack([OffSetSignal, np.tile(Cycle, (CycleNum, 1))])  # Record the overall signal of each demand level

            ControlVector = np.vstack([ControlVector, CycleList[:Time_SignalPeriod[j], :]])  # Integrate the offset thread

        Control[signal['Restricted'][0], :] = ControlVector[:, 0]
        Control[signal['Restricted'][1], :] = ControlVector[:, 1]

    Control = np.ones((LenLink, TotalTimeStep))  # Initialization
    Signal.apply(offset, 1)

    return Control
