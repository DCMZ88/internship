# Neural Networks
##  Table of Contents
- [Structure of a neural network](#structure-of-a-neural-network)
- [Components of a neural network](#components-of-neural-network)
- [Activation Functions](#activation-functions)
## What is a neural network 
A neural network is a type of machine learning model inspired by the structure and function of the human brain. It's designed to recognize patterns and make predictions by learning from data.
## Structure of a neural network 
- ### Input layer 
- ### Hidden layer
- ### Output layer
## Components of neural networK
- **Neurons** : Basic building blocks of a neural network where each neuron proccesses an input and produces an output
- **Layers** : Consists of neurons
- **Weights** : Each Connection between neurons has an associated weight $w_i$, which determines the strength of a connection
- **Activation Functions** : Functions applied to the output of each neuron to introduce non-linearity (e.g ReLu, Sigmoid )

## Activation Functions 
### ReLu ( Rectified Linear Unit )
- Most commonly used activation function for hidden layers

**When and why do we use ReLU**
- In deep learning (i.e CNNs, and FCNs )
- Introduces non-linearity into the network, allowing it to learn more complex functions and representations ( Hidden Layer )
- Output a positive number ( Output Layer )
- Computationally faster than other activation functions such as sigmoid

**How it works**

<img src="https://github.com/user-attachments/assets/f675663f-d983-41c0-ac63-9911e5705976" alt="relu" width="500"/>

where $z$ is the weighted sum of the inputs of the current layer plus a bias term 

- Each neuron in the current layer computes a weighted sum of its inputs (the activations from the previous layer) plus a bias term

$$ z = \sum_{i=1}^{n} w_i \cdot a_i + b $$

- The activation function $a=R(z)=max(0,z)$ is the applied to $z$ to produce the output activation $a$ which would then be passed on to the next layer




    
