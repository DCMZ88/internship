![image](https://github.com/user-attachments/assets/4abf9cd4-d333-47b5-8d1a-e2090899ca7a)# Intro to Machine Learning 
**Definition**: "Field of study that gives computers the ability to learn without being explicitly programmed." Arthur Samuel (1959)
## Contents
- [Types of Machine Learning](#types-of-machine-learning)
- [Linear Regression](#linear-regression)
   - [Cost Function](#cost-function)
   - [Gradient Descent(Optimization)](#Gradient-Descent)
   - [Challenges](#Challenges) 
- [Logistic Regression](#logistic-regression)
   - [Cost Function](#cost-function) 
## Types of Machine Learning 
### Supervised Learning (Main focus for internship)
- Algorithms which are trained on labelled data where each output has an associated output label 
- Applications : Classification , Regression
- Algorithms : Linear Regression , Neural Networks , Support Vector Machines  
### Unsupervised Learning
- Algorithms that are traine on unlabelled data so as to find patterns and relationships in the data itself 
- Applications : Clustering,Dimensionality Reduction
### Reinforcement Learning 
- Algorithms that learns by interactiing with an environment and receiving feedback in the form of rewards or punishments
- Applications : Robotics, Autonomous vehicles
## Process of ML
![Process](https://www.gptechblog.com/content/images/2023/06/02-machine-learning-process-2.png)
1. Define the problem
2. Preparing data for model
3. Define the model + loss function
4. Train the model (Minimise loss function)
5. Deployment 

## Supervised Learning (Algorithms)
### Linear Regression 
**Aim: Makes a predicted output from infinetly many possible numbers**

![Model for Linear Regression](https://github.com/user-attachments/assets/8ac75049-4400-4daa-bcd2-d52ec4ee08cd)

$w,b$ are parameters

$\vec{w}$, $\vec{x}$ are lists containing of each input feature

- Takes in $\vec{x}$  as an input and calculates the predicted value of $f_{\vec{w},b}(x)$ using the $\vec{w},b$ as parameters

#### Cost Function
Most Commonly Used function : **Mean Squared Error (MSE)** 

 ![MSE](https://github.com/user-attachments/assets/129fc8e9-1f12-4764-b06a-0cfd01d041be)
 
 Function:
- Calculates the difference between the predicted value of $\hat{y}$ and the actual value of $y$ for each data point 
- Sums all the differences and divides it by the number of examples, $n$

**Why use MSE?**
  : Because it only has one global minima , able to find the $(w,b)$ where the cost function is at the minimum

#### Optimizing the Cost Function
Now that we have defined the cost function through the use of MSE , what's left is to calculate the minimum cost function to find the best parameters $w,b$ for $f_{w,b}(x)$

**Optimization** 

For this case, we utilise **Gradient Descent**.

#### Gradient Descent

**Aim**
: To minimise cost function

![Algorithm(Gradient Descent](https://github.com/user-attachments/assets/775e6b31-70e9-453e-ae05-ff4ce6ff6549)

$\alpha$ is the learning rate and the derivative is the gradient of the curve at that point 

How it works:

1. Start with a random value of $w$ and $b$
2. Calculates the cost function using the new parameters ( Loop over all loss functions)
3. Calculates the gradient of the cost function
5. Updates the $w$ and $b$ (parameters) such that each new value of it calculated moves closer to a point where $J(w,b)$ is at a minimum
6. Repeats the process until a global minimum is reached ( i.e Cost function converges to 0 )
   
   ( Converges to a minimum )

The derivative eventually converges to 0 as at the minimum, gradient = 0, leaving us with the optimal values of $w$ & $b$

### Challenges 

**Vectorization**

For each iteration in gradient descent, each cost function has to loop over all the loss functions, making it computationally costly if data set is too large.

Thus it is advised to vectorize the equation using NumPy.

**Feature Scaling**

As the input features might be very large or very small, we want to ensure that these features helps the algorithm to converge faster as the algorithm is sensitive to the scale of the features.

Thus for this case, we aim to to scale the features to about $-1 \leq x \leq 1$ for each feature of $x$ using normalization.

**Normalization**

![Normalization Equation](https://github.com/user-attachments/assets/cb5263fa-ff4b-43db-87b1-c9a1b44ccbef)

**Feature Engineering**

Linear regression can take in many features $x_i$, however all of these features may not be meaningful to the model. Feature Engineering enables to features to be combined and transformed to better crafted features, enabling the model to capture underlying patterns in data more effectively, enhancing performance and accuracy.

( E.g , $x_1$ = breadth , $x_2$ = width can be combined into $x_1 \times x_2 = x_3 (area)$ )

### Logistic Regression 

**Aim** : Binary Classification ( Ouputs either a "1" or "0" )

![Logistics Regression Model](https://github.com/user-attachments/assets/c5303387-351c-4e44-873c-935a72317895)

$\vec{w},b$ are parameters 

Utilizes the sigmoid function $\sigma(z)$ to either output a "1" or "0"

#### Cost Function 
For this case, we utilize the binary cross-entropy loss function 

![Cost Function](https://github.com/user-attachments/assets/075f213b-aebb-4959-a7b2-0b34ce2ede8a)

Function:

- Difference between the predicted probabilities and the actual binary outcomes across the entire dataset. 
- For $y_i = 1$, as $\hat{y_i}$ converges to 1, then the loss function would converge to 0, vice versa.

where $y_i$ is the actual output and $\hat{y_i}$ is the predicted output.

<img src="https://github.com/user-attachments/assets/abf72d84-6cfc-4154-b988-49f7f4dd498c" alt="Graph" width="500"/>

- Huge penalty if the predicted output varies too much from the actual output (i.e $\hat{y_i}$ = 0.1, $y_i = 1$ , this causes the loss function to be huge)(Pink Curve)


**Optimization** 

For this case, we also utilise [Gradient Descent](#Gradient-Descent) in determining the minimum cost function which gives us the best parameters for $w,b$.























