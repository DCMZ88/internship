# Neural Networks
##  Table of Contents
- [Structure of a neural network](#structure-of-a-neural-network)
- [Components of a neural network](#components-of-neural-network)
- [Activation Functions](#activation-functions)
  - [ReLU Function](#relu-rectified-linear-unit)
  - [Sigmoid Function](#sigmoid-function)
- [Process of neural network](#how-a-neural-network-works)
  - [Types of optimization](#types-of-optimization-functions)
  - [Early Stoppage](#early-stoppage)
- [Convolution Neural Networks(CNN)](#convolutional-neural-network)
    - [Convolution Layer](#convolution-layer)
    - [Pooling Layer](#pooling-layer)
- [Transfer Learning](#transfer-learning)
## What is a neural network 
A neural network is a type of machine learning model inspired by the structure and function of the human brain. It's designed to recognize patterns and make predictions by learning from data.
## Structure of a neural network 
- ### Input layer 
- ### Hidden layer
- ### Output layer
    - Depends on the use of the neural network
        - Binary Classification ( Sigmoid Function )
        - Multi-class Classification ( Softmax Function )
        - Non-negative outputs ( ReLU )
        - Range of continous values ( Linear Activation Function a.k.a No Activation )
## Components of Neural Network
- **Neurons** : Basic building blocks of a neural network where each neuron proccesses an input and produces an output
- **Layers** : Consists of neurons
- **Weights** : Each Connection between neurons has an associated weight $w_i$, which determines the strength of a connection
- **Activation Functions** : Functions applied after output of each neuron in each layer to introduce non-linearity (e.g ReLu, Sigmoid )

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
    - **Activation** : Each neuron in the hidden layers calculates a weighted sum of its inputs, applies an activation function (e.g., ReLU, sigmoid), these activations are then combined into a single vector $a_i$, where $i$ is the $i$th layer of the network , and then is passed into the next layer 
    - **Output** : The output layer processes the input from the last hidden layer and produces the final prediction.
2. **Loss Calculation** : After each iteration (i.e after each batch is passed through the whole NN ), the loss between the predicted output $\hat{y}$ and the actual target values $y$ is calculated using a loss function ( e.g Mean-Squared Error(MSE), Cross-Entropy )
3. **Backward Propagation**:
    - Using optimization functions such as Gradient Descent, calculate gradients of the loss with respect to the weights in each layer.
    - The weights are then subsequently updated for each layer to minimise the loss 
4. **Iteration**
    - Repeats the process for each batch size and update weights using backpropagation
    - Repeats the process for the whole training data ( also known as one epoch ) until the loss function converges to a satisfactory value
### Types of Optimization Functions
- **Stochastic Gradient Descent** : Updates the parameters after every training example individually by selecting a random point on each example ( Recommended )
- **Mini-batch Gradient Descent** : Updates the parameters after a small batch of training examples
- **Batch Gradient Descent** : Updates the parameters after running through the whole training dataset
- **ADAM (Adaptive Movement Estimation)** : uses adaptive learning rates for each parameter, adjusting them individually based on the estimated first and second moments (mean and uncentered variance) of the gradients.

### Early Stoppage
As we train the model using the training data set, average loss will keep decreasing for each iteration of gradient descent, converging to a minimum. However, this may cause the model to overfit to the training data.

Hence, to avoid this, we use the validation data set to train our model, instead of optimizing to convergence, we optimize until validation loss stops improving. 

- Pros:
  - Able to check validation loss after every iteration
  - Ensures that it performs better on unseen and real-world data
  - Saves computational cost as iterating over each gradient descent until convergence is costly

<img src="https://github.com/user-attachments/assets/acf962a3-2c83-45ac-8398-fc5ff9e38e1a" alt="early stoppage" width="500"/>

As seen from the figure, the average loss $J_{train}$ converges until a minimum for every iteration whereas the loss $J_{validation}$ converges until a certain point before it starts to increase. 

Optimally, we would want to stop the optimization when $J_{validation}$ starts to degrade, achiveing the optimal loss and thus best parameters that minimise the cost function the most.


## Convolutional Neural Network

**Simplified structure of CNN**

<img src="https://github.com/user-attachments/assets/c43b9921-fecb-4d43-90e1-bd1bfebbcd23" alt="simple cnn" width="500"/>

**What is a Convolutional Neural Network?**

A Convolutional Neural Network (CNN) is a type of deep learning algorithm that is particularly effective for processing data with a grid-like topology, such as images.

### Structure of a CNN 

<img src="https://github.com/user-attachments/assets/f3a2b66a-fd08-422c-8393-8c2fa23c7637" alt="CNN" width="500"/>

- Convolution Layer
- Activation Functions
- Pooling Layers
- Flattening layer
- Fully Connected Layer
- Output Layer



### Convolution Layer

<img src="https://github.com/user-attachments/assets/82f59efd-e429-4339-a96e-cb14882625a4" alt="convolution" width="500"/>

**How it works**:
- Initialises random variables for the filters/kernels in the convolutional layer
- During forward propagation , filters are applied to the input data, producing feature maps ( where the features are stored )
  - Imagine it as matching shapes of the filter to each pixel of the image and thus producing a map where the features are most prominent where the shapes on the image matches the filter.
- The loss is then calculated and the values of these filters are then updated through optimzation functions like Gradient Descent
- Through multiple iterations, these filters are then adjusted to better detect patterns and shapes in the data
- Through this iterative process, the filters evolve from random noise to meaningful detectors of features such as edges, textures, and shapes.
- Over time, the network learns to extract increasingly complex and abstract features that are crucial for the given task.


## Transfer Learning 
**Aim** : Transfer knowledge from pre-trained models to new models so as to reduce training time and improve performance

How it works:
- Feature Extraction: Transfers the weights and parameters of the lower layers of the filter from a pretrained-model to a new model

<img src="https://github.com/user-attachments/assets/7091ae56-2d71-456f-90be-b7e7aa2d3250" alt="transfer learning" width="500"/>

As seen from the image, the convoutional layers from the model using ImageNet are transferred to another model to process confocal images.

- Fine-tuning : After transfering the architecture of a pretrained model, we would then need to fine-tune some layers and train the model to suit our specific task as shown in the image where the model is fine-tuned further to be able to analyse the confocal images.

**Why use transfer learning?**
- It reduces the time to train the model as some of its parameters have already been optimized
- Reduces data needed to train the model
- Improve performance of the model as these pre-trained models have already learned useful features from large datasets.












    
