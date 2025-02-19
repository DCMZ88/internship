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
![Model for Linear Regression](https://github.com/user-attachments/assets/520a88a6-457b-4fdf-b84b-ac6081447c77)
*w,b* are parameters 

- Takes in x<sup>(i)</sup>  as an input and calculates the predicted value of y (F<sub>w,b</sub>) using the *w,b* as parameters

#### Cost Function
Most Commonly Used function : **Mean Squared Error (MSE)** 
 ![MSE](https://github.com/user-attachments/assets/129fc8e9-1f12-4764-b06a-0cfd01d041be)
 
 Function:
- Calculates the difference between the predicted value of y and the actual value of y
- Sums all the differences and divides it by the number of examples, *n*
**Why use MSE?**
  : Because it only has one global minima , able to find the *(w,b)* where the cost function is at the minimum

#### Optimizing the Cost Function
Now that we have defined the cost function through the use of MSE , what's left is to calculate the minimum cost function to find the best parameters *w,b* for F<sub>w,b</sub>.

**Optimization** 

For this case, we utilise **Gradient Descent**.

**Aim**
: To minimise cost function

![Algorithm(Gradient Descent](https://github.com/user-attachments/assets/775e6b31-70e9-453e-ae05-ff4ce6ff6549)

How it works:

1. Start with a random value of w and b
2. Updates the W and B such that each new value of it calculated moves closer to a point where J(W) is at a minimum
   ( Converges to a minimum )

Alpha is the learning rate and the derivative is the gradient of the curve at that point 

The derivative eventually converges to 0 as at the minimum, gradient = 0, leaving us with the optimal values of *w* & *b*







