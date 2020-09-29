import numpy as np


def ControlVector_Webster(LenLink, SignalControl, TotalTimeStep, Time_SignalPeriod, LostTime):
    '''
    Control vector
    binary: 0 - RED 1-GREEN
    each link should only be associated with one Control vector
    Generic coding of Webster plan with offsets
    '''
    Control = np.ones((LenLink, TotalTimeStep))  # Initialization

    for sig in SignalControl:
        ControlVector = np.array([]).reshape(-1, 2)  # Initialize the Control vector of each signal

        for j in range(sig.GreenSplit.shape[0]):
            CycleTime = sum(sig.GreenSplit[j, :]) + 2 * LostTime  # Calculate Cycle time

            Cycle = np.zeros((CycleTime, 2))  # Initialize signal
            Cycle[:sig.GreenSplit[j, 0], 0] = 1  # First stage with green
            Cycle[sig.GreenSplit[j, 0] + LostTime:sig.GreenSplit[j, 0] + LostTime + sig.GreenSplit[j, 1], 1] = 1  # Second stage with green

            # Add the thread in the head of signal as offsets
            OffSetSignal = Cycle[-sig.Offset % CycleTime:]

            CycleNum = Time_SignalPeriod[j] // CycleTime + 1  # Calculate the number of cycles

            CycleList = np.vstack([OffSetSignal, np.tile(Cycle, (CycleNum, 1))])  # Record the overall signal of each demand level

            # StartTime = (CycleTime - sig.Offset % CycleTime) % CycleTime  # determine the start time including the offset period
            # StartTime = OffSetSignal.shape[0] - sig.Offset  # determine the start time including the offset period
            # EndTime = StartTime + Time_SignalPeriod[j]

            ControlVector = np.vstack([ControlVector, CycleList[:Time_SignalPeriod[j], :]])  # Integrate the offset thread

        Control[sig.Restricted[0], :] = ControlVector[:, 0]
        Control[sig.Restricted[1], :] = ControlVector[:, 1]

    # np.savetxt('control.csv', Control, fmt='%d', delimiter=',')
    return Control
