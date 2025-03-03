# Neural Networks
##  Table of Contents
- [Structure of a neural network](#structure-of-a-neural-network)
- [Components of a neural network](#components-of-neural-network)
- [Activation Functions](#activation-functions)
  - [ReLU Function](#relu--rectified-linear-unit-)
  - [Sigmoid Function](#sigmoid-function)
- [Process of neural network](#how-a-neural-network-works)
  - [Types of optimization](#types-of-optimization-functions)
  - [Early Stoppage](#early-stoppage)
- [Convolution Neural Networks(CNN)](#convolutional-neural-network)
    - [Convolution Layer](#convolution-layer)
        - [Activation Function](#activation-function)
    - [Pooling Layer](#pooling-layer)
        - [Flattening](#flattening)
    - [Fully Connected Layer](#fully-connected-layer)
        - [Output Layer](#output-layer-1)
- [Transfer Learning](#transfer-learning)
- [Transformers](#transformers)
  - [Sequence Encoder](#sequence-encoder)
  - [Decoder](#decoder)
## What is a neural network 
A neural network is a type of machine learning model inspired by the structure and function of the human brain. It's designed to recognize patterns and make predictions by learning from data.
## Structure of a neural network 
- ### Input layer 
- ### Hidden layer
  - Made up fully connected layers ( linear layers ) which are usually paired with activation functions to introduce non-linearity.
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
### ReLU ( Rectified Linear Unit )
- Most commonly used activation function for hidden layers

**When and why do we use ReLU**
- In deep learning (i.e CNNs, and FCNs )
- Introduces non-linearity into the network, allowing it to learn more complex functions and representations ( Hidden Layer )
- Output a positive number ( Output Layer )
- Computationally faster than other activation functions such as sigmoid

**How it works**

<img src="https://github.com/user-attachments/assets/ebb34d2a-34f0-4a15-928a-237f4ecd6179" alt="relu" width="500"/>

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
    - **Activation** : Each neuron in the hidden layers calculates a weighted sum of its inputs ( Linear Layer ), applies an activation function (e.g., ReLU, sigmoid), these activations are then combined into a single vector $a_i$, where $i$ is the $i$th layer of the network , and then is passed into the next layer 
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

**Parameters**
- Filter Size 
- Fiter Stride ( How many pixels does the filter go over in one step )  
- Volume ( The feature maps from each individual chanel i.e RGB )

#### Activation Function 
- Applied to the output of the convolutional layer to introduce non-linearity
- Examples: ReLU, Sigmoid , TanH

### Pooling Layer
**Aim** : To reduce computational complexity
**How it works**
- Types of Pooling
  - Max Pooling : Takes the maximum value from each region covered by the filter.
  - Average Pooling :Takes the average value from each region covered by the filter.
  - Min Pooling : Takes the minimum value from each region covered by the filter. ( Less Commonly Used )

<img src="https://github.com/user-attachments/assets/faac794d-55a3-4223-85c9-710c8c28c190" alt="pooling" width="500"/>

#### Flattening 
**Aim** : Prepare the data for the Fully Connected Layer
**How it works** 
- Converts the multi-dimensional feature maps into a one-dimensional vector

### Fully Connected Layer
**Aim**: Combines the features learned by the convolutional and pooling layers to make predictions or classifications.
**How it works**
- Works exactly like a [Neural Network](https://github.com/DCMZ88/internship/blob/main/Week%203/MachineLearning.md#neural-networks)
  - Initialization
  - Forward Propagation
  - Prediction
  - Loss Function
  - Back Propagation
  - Iteration
#### Output Layer 
**Aim** : The final output, such as class probabilities or regression values, depending on the task.
**How it works**
- Depending on the activation function used to output a value or probability


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


## Transformers

**What are transformers**


### Structure of a transformer 

<img src="https://github.com/user-attachments/assets/e40ddee7-0ade-4dcd-9583-d2fdd98b55b0" width="500"/> 

#### Sequence Encoder
1. Initial Embeddings
2. Multi-Head Attention
3. Skip Connection ( Residual Connection )
4. Feed Forward network
5. Skip Connection 

**Aim** : To understand the context of the sentence 

- Initially, the input embeddings $c_k$ are fed through the multi-head attention to output the new embeddings $\tilde{c_k}$. 
- Before passing it through the feed forward network, the inital embeddings $c_k$ is added to the new embedding $\tilde{c_k}$ (i.e Adding the original word to its context )
- This is so as to ensure that the new token preserves some of its orginal features and to avoid degradation ( overfitting )
- This process is then repeated for $x$ number of times
### Initial Embeddings 
- **Token Embeddings** : Breaks down the sentences into tokens (e.g. words,characters) and intialises these tokens as vectors 
- **Positional Encoding**: The position of the token in the sentence is also embedded as a vector into the token

### Self-attention Block
**Aim** : Pass information back and forth about the other tokens in the sentence ( i.e For each token, how important is the other tokens in relation to it , gives us the context of the tokens based on its surrounding tokens )

- #### Query Vector 
  The query vector represents what the current token (or position in the sequence) is trying to understand or retrieve information about from the entire sequence.
   (i.e Asking other tokens for information )
  
  $\vec{Q_i} = W_Q \times \vec{E_i}$

  where $\vec{E_i}$ is the Initial Embeddings and $W_Q$ is the learned weight matrix for queries

- Key Vector ( $K_i$ ): Represents the information available in each token of the sequence. Acts like labels/description for the token. Similar to query vectors, key vectors are generated by multiplying the input embeddings with a weight matrix, $W_k$. 
  (i.e Answers the queries of the tokens / Finds the relevance between the answers and the question )  

- Value Vector ( $V_i$ ): Contains the actual information to be aggregated ( The answers that will be used ) . Value vectors are generated by multiplying the input embeddings with a weight matrix $W_V$.

#### Attention Score Calculation 

<img src="https://th.bing.com/th/id/OIP.hG-cte_k98lsp7b8G1rMJQHaBm?rs=1&pid=ImgDetMain" width="500"/>

where $d_k$ is the dimension of the query space

Softmax is used in this case to normalize the values of the keys and query
 - The higher the probability, the stronger the relation

The results is a change in the embeddings $\Delta{\vec\{{E_i}^x}}$ produced by the value vector and the product of the key vectors and the query vectors.

For better visualisation

<img src="https://github.com/user-attachments/assets/a1238c76-d84e-4364-b279-42a5ca7ece6d" width="500"/>

where $c_k$ represents the input embeddings 


### Multi-headed Attention

Made up of many self-attention blocks which are calculated in parallel

The result of each attention block of each token is then concatenated to produce the total change $\Delta{\vec{E_i}}$ which has the same dimension as the original embedding.
( So if theres 10 tokens, there will be 10 different concatenated vectors )

This change ( $\Delta{\vec\{E_i}}$ ) is then added to the original embedding $\vec{E_i}$ to find the corresponding value of the embedding ( These embeddings contain information such as context of the other tokens in the sentence e.t.c )

The size of the output vector from each attention head can be represented by $h \times d_{head} = E $

whereby $h$ is the number of attention heads and $d_{head}$ is the dimension of the attention head and E is the size of the original embedding 

**HENCE**

- More attention heads means that the dimensions of the information ( $d_head$ ) that can be represented is lesser ( i.e less complex information and patterns can be represented )
- Less attention heads means that performance may not be optimal and information may not be as well-learnt as those in more attention heads. 

### Skip Connection 

-  The original embedding of the token $c_k$ is then added back into the vector so as to preserve original information
-  The vector is then normalized and then passed through the Multi-Layer Perceptron ( MLP )

### Multi-Layer Perceptron 

- The vector is passed through a MLP ( Feed Forward Layer ) which is a [neural network](#neural-network-1) which allows the tokens to learn
  complex patterns and relationship
#### Structure of MLP in a encoder 
  - **Input Layer**
  - **First Linear Layer**
  - **Activation Function**
  - **Second Linear Layer**
    
#### First Linear Layer
  - This increases the dimensionality of the vectors so as to capture more features

#### Activation Function
  - Applies activation functions such as ReLU to introduce non-linearity to the vector to allow it to capture more complex relationships in the data

#### Second Linear Layer
  - Projects the output from the activation function back to the original dimension ( the original input to the MLP )
#### Skip Connection
  - The input embedding is then added back to the output vector to help preserve original information
  - This vector is then normalized.

### Decoder
1. Masked Multi-Head Attention
2. Multi-Head Attention
3. Feed Forward Layer
4. Linear Layer
5. Softmax

**How it works**
- Initially, the input token for the decoder is a special [START] token which gets input into the masked multi-head attention blocl
- The output is passed through a skip connection before passing it into the cross-attention block
- This multi-head attention would utilise the output from the encoder to generate a contextualised vector for the tokens in the sentence
- This output is passed through a skip connection again before passing through the feed forward layer
- The Multi-Layer Perceptron then further refines the token before generating its final predictions
- The embedddings are then passed through a skip connection
- Finally, the output is passed through a linear layer to reduce its dimensionality to match the size of the vocabulary and afterwards a softmax layer to output its predictions for the next word in probabilities
- The token with the highest probability is then generated is then fed back to the encoder in order to generate the next token for the sentence
- This process repeats until all the tokens have been predicted

### Masked Multi-Head Attention 
- Works like the [Multi-Head Attention](#multi-headed-attention), consists of many self-attention heads
 
**Masking** ( Only for decoder ) 

**Aim**: To ensure that the later words do not influence the previous words when calculating the attention score

Even though we are only using the previous word as input , but this word contains the weights of all the other tokens and thus we have to mask these weights of the tokens that are after the current predicted token. 

- We initialise all these words to negative infinity such that when softmax is applied, the values are close to 0.

### Multi-Head Attention ( Cross Attention )

For this multi-head attention, we utilise the output values from the encoder block ( $K,V$ ) as input to the self-attention block so as to 











    
