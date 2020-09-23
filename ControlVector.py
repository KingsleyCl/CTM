import numpy as np
import UrbanStreet as us


def ControlVector_Webster(LenLink, TotalTimeStep, SignalControl, Time_SignalPeriod, LostTime):
    '''
    control vector
    binary: 0 - RED 1-GREEN
    each link should only be associated with one control vector
    Generic coding of Webster plan with offsets
    '''
    control = np.ones((LenLink, TotalTimeStep))  # Initialization

    for sig in SignalControl:
        ControlVector = np.array([]).reshape(-1, 2)  # Initialize the control vector of each signal

        for j in range(sig.GreenSplit.shape[0]):
            CycleTime = sum(sig.GreenSplit[j, :]) + 2 * LostTime  # Calculate Cycle time

            Cycle = np.zeros((CycleTime, 2))  # Initialize signal
            Cycle[:sig.GreenSplit[j, 0], 0] = 1  # First stage with green
            Cycle[sig.GreenSplit[j, 0] + LostTime:sig.GreenSplit[j, 0] + LostTime + sig.GreenSplit[j, 1], 1] = 1  # Second stage with green

            # Add the thread in the head of signal as offsets
            # OffSetSignal = np.tile(Cycle, (4, 1))  # Use 4 cycles to be the thread of the offset

            Cycle_Num = Time_SignalPeriod[j] // CycleTime + 1  # Calculate the number of cycles

            CycleList = np.tile(Cycle, (Cycle_Num, 1))  # Record the overall signal of each demand level
            # CycleList = np.hstack([OffSetSignal, np.tile(Cycle, (Cycle_Num, 1))])  # Record the overall signal of each demand level

            StartTime = CycleTime - sig.Offset % CycleTime  # determine the start time including the offset period
            # StartTime = OffSetSignal.shape[0] - sig.Offset  # determine the start time including the offset period
            EndTime = StartTime + Time_SignalPeriod[j]

            ControlVector = np.vstack([ControlVector, CycleList[StartTime:EndTime, :]])  # Integrate the offset thread

        control[sig.Restricted[0], :] = ControlVector[:, 0]
        control[sig.Restricted[1], :] = ControlVector[:, 1]

    return control


SignalControl = []
SignalControl.append(us.signal(8, 43, [14, 20], [24, 35, 56, 82, 24, 35]))
SignalControl.append(us.signal(9, 93, [15, 20], [25, 34, 59, 79, 25, 34]))
control = ControlVector_Webster(21, 3600, SignalControl, [900, 1800, 900], 5)
np.savetxt('result.csv', control, delimiter=',')
