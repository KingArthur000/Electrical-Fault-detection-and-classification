# Electrical-Fault-detection-and-classification
A sample power system was modelled using MATLAB simulink and all six types of faults was introduced into the transmission line of the power system. A ML classifier using 8 types model was implemented using sklearn and appropriate model was selected as the end model for each problem.

## Problem Statement:
The main objective of our Project is to Detect and classify  the faults on electrical power transmission lines using artificial neural networks. The three phase currents and voltages of one end are taken as inputs in the proposed scheme. The feed forward neural network along with back propagation algorithm has been employed for detection and classification of the fault for analysis of each of the three phases involved in the process. 
A detailed analysis of 8 different models with varying numbers of hidden layers has been performed to validate the choice of the neural network. 
The simulation results concluded that the present method based on the neural network is efficient in detecting and classifying the faults on transmission lines with satisfactory performances. The different faults are simulated with different parameters to check the versatility of the method. The various simulations and analysis of signals is done in the MATLAB® environment.

## Introduction:
Transmission line is the most important part of the power system. The requirement of power and its allegiance has grown up exponentially over the modern era, and the major role of a transmission line is to transmit electric power from the source area to the distribution network. The electrical power system consists of so many different complex dynamic and interacting elements, which are always prone to disturbance or an electrical fault.
The use of high capacity electrical generating power plants and concept of grid, i.e. synchronized electrical power plants and geographical displaced grids, required fault detection and operation of protection equipment in minimum possible time so that the power system can remain in stable condition. The faults on electrical power system transmission lines are supposed to be first detected and then be classified correctly and should be cleared in the least possible time. The protection system used for a transmission line can also be used to initiate the other relays to protect the power system from outages. A good fault detection system provides an effective, reliable, fast and secure way of a relaying operation. The application of a pattern recognition technique could be useful in discriminating against faulty and healthy electrical power systems. It also enables us to differentiate among three phases which phase of a three phase power system is experiencing a fault. 
The artificial neural networks (ANNs) are very powerful in identifying the faulty pattern and classification of fault by pattern recognition. An efficient and reliable protection method should be capable of performing more than satisfactory under various system operating conditions and different electrical network parameters. As far as ANNs are considered they exhibit excellent qualities such as normalization and generalization capability, immunity to noise, robustness and fault tolerance. Therefore, the declaration of fault made by ANN-based fault detection method should not be affected seriously by variations in various power system parameters.
The various electrical transient system faults are modelled, simulated and Various  ANN based algorithms are developed for recognition of these faulty patterns. The performance of the proposed algorithm is evaluated by simulating the various types of fault and the results obtained are encouraging.

## Power System:
We have modelled a power system in MATLAB to simulate fault analysis.The system consists of 4 generators of 11 × 103 V, each pair located at each end of the transmission line, with transformers in between to simulate and study the various faults at the midpoint of  the transmission line.

MATLAB simulation of power system

<img src="/pics/power system.png" alt="Power system"/>

## Dataset:
We simulate the circuit under normal conditions as well as under various fault conditions. We then collect and save the measured Line Voltages and line currents at the output side of the power system. we simulated nearly 12000 data points and then the data is labelled.

### Simulated faults :

Example-LG Fault
<img src="/pics/lg.png" alt="LG"/>

All faults
<img src="/pics/all.png" alt="ALL faults"/>

Switching Time for each fault for the above graph
<img src="/pics/fault.jpg" alt="FAult timings"/>

## Fault detection:
For fault detection, the collected data is labelled as 0 or 1, where 0 is for no fault and 1 if fault is present.
<img src="/pics/detect.png" alt="Detection problem"/> 

Inputs-[Ia,Ib,Ic,Va,Vb,Vc]
Outputs- 0(No fault) or 1(Fault is present)

### Fault classification :
For fault detection the same line voltages and line currents are collected for various faults. The output class consists of 4 binary variables each for Ground, Phase A, Phase B and Phase C respectively.

Snapshot of Classification dataset for AG fault
<img src="/pics/class.png" alt="Classification problem"/>

Inputs- [Ia,Ib,Ic,Va,Vb,Vc]                
Outputs[G C B A]- 
Examples :
                        [0 0 0 0] -No Fault
                        [1 0 0 1]- LG fault (Between Phase A and Gnd)
                        [0 0 1 1]- LL fault (Between Phase A and Phase B)
                        [1 0 1 1]- LLG Fault (Between Phases A,B and ground)
                        [0 1 1 1]-LLL Fault(Between all three phases)
                        [1 1 1 1]- LLLG fault( Three phase symmetrical fault)
