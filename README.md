# Attention based Time Series Analysis Drill Bit

The objective of this study is to perform a comprehensive Drill Bit Analysis aimed at predicting the Output Speed and Weight on the Drill Bit based on the Input Velocity and Input Torque applied to the drill bit. Both experimental and simulational data are used to develop a robust predictive model. Efforts are made to align the predictions for the simulational model with experimental results, leveraging a data-driven approach using Recurrent Neural Networks (RNN).

## Problem Description
In drilling operations, the performance and efficiency of the drill bit are heavily influenced by the interaction between input parameters such as velocity and torque and the resulting output parameters, i.e., speed and weight on the bit. Traditionally, these parameters have been experimentally determined. However, the goal here is to develop a predictive framework that utilizes a simulational model to forecast the Output Speed and Weight on the Drill Bit based on the Input Velocity and Torque using the same predictive model employed for experimental data.

## Simulational Model
A computational drill bit model was developed to simulate various operational scenarios. This model replicates real-world drilling conditions and allows for controlled variation of input parameters such as velocity and torque. The simulation generates corresponding output parameters, which include:

**Output Speed:** The rotational speed achieved by the drill bit.
**Weight on Bit**: The force exerted on the bit during the drilling process.
The simulation results serve as synthetic data for validating and improving the predictive model.

## Predictive Model Development
The predictive model is designed using Recurrent Neural Networks (RNN), a machine learning architecture well-suited for time-series data, given the continuous nature of drill operations and the relationship between the input and output parameters.

## Experimental Data Prediction 
Real-world experiments are conducted to gather data on the input velocity and torque applied to the drill bit and the resulting output speed and weight on the bit. This data is essential for training the RNN model.
![image](https://github.com/user-attachments/assets/d6754b79-136e-4821-a2c9-71d8edcefb0b)

![image](https://github.com/user-attachments/assets/e34d2607-de9c-43ff-a774-d7f02f8e4e7e)

## Simulational Data Generation
The simulational model provides additional data points, simulating various conditions and helping augment the experimental dataset.

## RNN Training
Both experimental and simulational datasets are used to train the RNN model. The model takes as input the Input Velocity and Torque and predicts the Output Speed and Weight on Bit. The training process involves the following:
Preprocessing the data to normalize input parameters.
Using a sequence of historical input values to predict future output values, leveraging the time-dependent nature of the problem.
Tuning the RNN model’s hyperparameters (such as number of layers, learning rate, etc.) to optimize prediction accuracy.
Validation and Testing: The trained RNN model is validated using unseen experimental and simulational data to ensure consistency in the predictions for both scenarios. Cross-validation techniques are applied to prevent overfitting.
