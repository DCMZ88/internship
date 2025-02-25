# Neural Networks
##  Table of Contents
- [Structure of a neural network](#structure-of-a-neural-network)
- [Components of a neural network](#components-of-neural-network)
- [Activation Functions](#activation-functions)
  - [ReLu Function](#relu)
  - [Sigmoid Function](#sigmoid-function)
- [Process of neural network](#how-a-neural-network-works)
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

### Sigmoid Function
- Same as the function used in [Logistic Regression](https://github.com/DCMZ88/internship/blob/main/Week%203/MachineLearning.md#logistic-regression)
**When and why do we use sigmoid**
- Binary classification ( Output layer )
- ReLU still preferred method for hidden layers in deep learning

**How it works**

<img src="https://github.com/user-attachments/assets/09100a20-63ec-4b2f-af88-7ef22a27ff9a" alt="sigmoid" width="500"/> 

where $t$ is the weighted sum of the inputs of the current layer plus a bias term 

- The activation function $sig(t)$ outputs a probability from 0 to 1


## How a neural network works 
1. **Forward Propagation**
    - **Input** : The data is being fed into the input layer
    - **Activation** : Each neuron in the hidden layers calculates a weighted sum of its inputs, applies an activation function (e.g., ReLU, sigmoid), and passes the result to the next layer.
    - **Output** : The output layer processes the input from the last hidden layer and produces the final prediction.
2. **Loss Calculation** : After each iteration (i.e after each batch is passed through the whole NN ), the loss between the predicted output $\hat{y}$ and the actual target values $y$ is calculated using a loss function ( e.g Mean-Squared Error(MSE), Cross-Entropy )
3. **Backward Propagation**:
    - Using optimization functions such as Gradient Descent, calculate gradients of the loss with respect to the weights in each layer.
    - The weights are then subsequently updated for each layer to minimise the loss 
4. **Iteration**
    - Repeats the process for each batch size and udate weights using backpropagation
    - Repeats the process for the whole training data ( also known as one epoch ) until the loss function converges to a satisfactory value










    
