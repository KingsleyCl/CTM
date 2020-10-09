# **CTM**
A cell transmission model based on python.

 **Required package**: numpy, pandas, xlrd, matplotlib
## **1. UrbanStreet**
### **config**(**file**: str) -> (List[node], List[link], List[signal])
Read the urban street data from an Excel file. The file contains sheets *Node*, *Link* and *SignalControl*. Argument *file* is the name of the file. In the file, if there are more than one value in a cell, separate it with a ';'.
## **2. ControlVector**
### **ControlVector_Webster**(**Signal**: List[signal], **LenLink**: int, **TotalTimeStep**: int, **Time_SignalPeriod**: List[int], **LostTime**: int) -> numpy.array[numpy.array[int]]
Generate a signal control matrix (LenLink * TotalTimeStep).
## **3.CTM**
### (1) **Slice**(**Link**: List[link], **Node**: List[node], **Signal**: List[signal], **dt**: int)
Split links into cells.
### (2) **CTM**(**Control**: numpy.array[numpy.array[int]], **Link**: List[link], **Node**: List[node], **dt**: int, **TotalTimeStep**: int) -> numpy.array[numpy.array[float]] * 3
Model operations are performed by traversing the elements.
### (3) **CTM_matrix**(**Control**: numpy.array[numpy.array[int]], **Link**: List[link], **Node**: List[node], **dt**: int, **TotalTimeStep**: int) -> numpy.array[numpy.array[float]] * 3
Model operations are performed by matrix operations. There will be warnings during execution, but it will be faster.