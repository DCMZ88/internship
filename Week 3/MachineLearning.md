# Intro to Machine Learning 
**Definition**: "Field of study that gives computers the ability to learn without being explicitly programmed." Arthur Samuel (1959)

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

However, for each iteration in gradient descent, each cost function has to loop over all the loss functions, making it computationally costly if data set is too large.







