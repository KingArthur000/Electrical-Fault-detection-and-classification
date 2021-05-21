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
                        
## Suitability of the selected Network model:
The models taken into consideration were mostly classification models, i.e logistics regression, KNN, Decision tree classifier, we also wanted to check the validity of using Neural Networks such as perceptron models hence the inclusion of MLP classifier. The currents and voltages of the power system are sinusoidal in nature hence we wanted to fit a polynomial curve of higher degree to determine its predictability, hence the addition of polynomial regression. Finally to check how the predicted output matches/echoes the original dataset linear regression model was employed.
As from the previous section we have selected the Decision tree classifier for the detection problem. Decision tree continuously splits the data into hierarchical structures where the two child data points can be grouped under a particular parent's characteristic. This is a supervised categorical learning algorithm. 

Hence the model can also be extended to a classification model, in the future developments of the project. 
Now for classification problem we have selected a polynomial regression model. As the name suggests the polynomial degree belongs to natural numbers. A test was conducted to select the appropriate degree for the regression model, 

<img src="/pics/poly.png" alt="poly deg selection"/>
The above curve depicts that the error is minimum for polynomial degree equal to 4. Hence we have selected the value and problem suits well for the test dataset from the parameters derived from the train dataset.

## RESULT:
We have performed prediction of faults using various models namely Logistic, Polynomial regression, MLPC, Naive Bayes, D-Tree, SVM, KNN. Through our extensive analysis, we have plotted the corresponding accuracies and errors for each of the models implemented for both the detection and classification problem

<img src="/pics/accdetect.png" alt="Detection problem"/>
<img src="/pics/classdetect.png" alt="Classification problem"/>
By these results we can see that for Detection of fault,  Decision Tree give the least error i.e 0.53 percentage, and for classification of fault polynomial regression has the least error i.e. 0.62 percentage. 
Hence the above models are considered for the corresponding problems and can well fit any real time data, due to the variety of data points present in the dataset.

## INFERENCE:
We have studied the application of artificial neural networks for the detection and classification of faults on a three phase transmission lines system. The method developed utilizes the three phase voltages and three phase currents as inputs to the neural networks. The results are shown for line to ground, line-to-line, double line-to-ground and symmetrical three phase faults. All the artificial neural networks studied here adopted the back-propagation neural network architecture. The simulation results obtained prove that the satisfactory performance has been achieved by some of the proposed neural networks and out of those the polynomial regression model for classification problem and Decision tree classifier for Detection problem are practically implementable. The importance of choosing the most appropriate ANN configuration, in order to get the best performance from the network, has been stressed upon in this project. 

Artificial neural networks are a reliable and effective method for an electrical power system transmission line fault classification and detection especially in view of the increasing dynamic connectivity of the modern electrical power transmission systems. The performance of an artificial neural network should be analyzed properly and particular neural network structure and learning algorithm before choosing it for a practical application. 

The scope of ANN is wide enough and can be explored more. The fault detection and classification can be made intelligent by nature by developing suitable intelligent techniques. This can be achieved if we have computers which can handle large amounts of data and take less time for calculations.
