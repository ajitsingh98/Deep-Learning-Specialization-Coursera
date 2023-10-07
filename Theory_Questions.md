# Interview Questions

Contents
---
1. [Neural Networks and Deep Learning](#neural-networks-and-deep-learning)
2. [Hyperparameter tuning Regularization and Optimization](#hyperparameter-tuning-regularization-and-optimization)
3. [Structuring Machine Learning Projects](#structuring-machine-learning-projects)
4. [Convolutional Neural Networks](#convolutional-neural-networks)
5. [Sequence Models](#sequence-models])


Questions
---
## Neural Networks and Deep Learning
1. What exactly a neural network depicts?
2. What is supervised learning? Give some example of it?
3. What type of data supervised learning demands?[Structures/Unstructured]
4. Give some example of Structured and Unstructured data?
5. In last 10 years there has been a lot of advancement in deep learning, list down reasons for it?
6. Draw schematic diagram of FFNNs, CNNs and RNNs?
7. Draw sigmoid function $f(x)$ and answer the following questions.
   1. Evaluate $\lim f(x)_{{x \to \infty}}$.
   2. Evaluate $\lim f(x)_{{x \to -\infty}}$.
   3. Evaluate $f(x)$ at $x=0$.
8. In what scenarios we should use logistic regression?
9. Why it is not appropriate to use linear regression for classification tasks?
10. Derive loss function in case of logistic regression.
11. Comment on loss function in case of following scenarios:
    1. $y_true = 1$
    2. $y_true = 0$
12. What is the relationship between cost function and loss function?
13. What does derivative a function depicts? Illustrate it using an appropriate diagram.
14. Find the first and second derivative of sigmoid function?
15. What are different techniques to train the parameters of a neural network?
16. How does gradient descent algorithm works?
17. Find the derivate of loss function $L(a, y)$ in case of logistic regression wrt its parameters($w, b$)?
18. Explain how vectorization implementation using `numpy` is more scalable than the non-vectorized implementation of ml algorithms?
19. Compare and contrast the vectorized and non-vectorized implementation of gradient descent in logistic regression?
20. How does boardcasting technique help in easy and efficient implementation of various algorithms?
21. Write general principles in broadcasting technique.
22. What are some drawbacks of broadcasting technique?
23. Write pseudocode for constructing a neural network?
24. Draw a neural network with 1 hidden layer and annotate its each layer?
25. Suppose you have a feed forward neural network with just 1 hidden layer and the units in input layer are 50 and in hidden layer there are 100 units and output layer has 10 units. Find out total number of trainable parameters in the whole network?
26. Write the all possible equations that we need to solve in one forward pass of following neural network?
27. Why do we need to use the non-linear activation layers in neural networks?
28. Prove the without non-linear actiavtion function, neural networks are nothing but a linear regression models.
29. Explain why the $hyperbolic tangent$ $\tanh$ is almost always better to use in place of $sigmoid$ activation function?
30. What are the downside of using $\tanh$ or $sigmoid$ functions?
31. Define **ReLU** activation function.
32. Plot the **ReLU** function and comment on its derivative at $x>0$, $x=0$ and $x<0$.
33. What is the issue with **ReLU** and how does we can overcome it limitations?
34. Draw schematic diagram of **Leaky ReLU** and comment on its derivative at $x>0$, $x=0$ and $x<0$.
35. In  which case the learning in new ral networks will be relatively faster [tanh\sigmoid\relu]?
36. Find out derivative of following activation functions:
    1. Sigmoid
    2. Tanh
    3. ReLU
    4. Leaky ReLU
37.  Implement gradient descent(forward pass/backward pass) for neural network with one hidden layer(see the below image).
38. Why do we need to initialize the parameters of neural networks beforehand?
39. What are different parameters initializing strategies?
40. What happens if you initialize neural network's weights to zero?
41. Why we should initialize the parameters at random?
42. What happens if you initialize the paramters of neural networks to too large or too small values?
43. Find the dimensions of weight and bias matrices of each layer in the given network?
44. Draw deep neural network with following hiddent layers
    1. 1 hidden layer
    2. 2 hidden layers
    3. 5 hidden layers
45. Why do we need deep representation of neural networks in case of complex probalems like image recognition, speech recognition system etc.?
46. What is the difference between parameters and hyper-parameters? List down some parameters and hyper-parameters in case of FFNN?
47. Draw the gradient flow diagram in case of DFNN?

## Hyperparameter tuning Regularization and Optimization

1. Explain with some examples how does the choice of hyper-parameters impact the overall performance of a model?
2.  What does it mean to tune the hyper-parameters of a model?
3.  Why do we need to segregate the data into Train/Test/Dev sets?
4.  What is cross-validation?
5.  How can we we use cross-validation to tune hyper-parameters?
6.  Why it is good to have test sets along with train and dev sets?
7.  Is it necessary to have sev and test sets from same distribution?
8.  What are different data splitting strategies?
9.  Define bias and variance of a learner?
10. Explain bias-variance tradeoff?
11. Comment on bias and variance for following models?
12. Explain over-fitting and under-fitting situations?
13. How does overfitting/underfitting are related with bias/variance?
14. How can we relate bias/variance with train/dev/test sets errors?
15. Suppose you are evaluating a neural network and you found the train error is 60% and the validation error is 70%. Explain this situations and also mentions the steps you can take to overcome this?
16. In model evaluation step you found out that your mode is performing quite well on train set but the performance is not so good on developement sets. Is it good or bad? Explain the setps you should take to takle this situation?
17. Can a neural network with 5 layers suffer from high bias?
18. Can a neural network with 2 layers suffer from high variance?
19. If a hypothesis is suffering from high bias, getting more training data can help?
20. How does getting more data can help in reducing varaince a model?
21. What does regularisation mean?
22. What is the impact of regularization on bias and variance of a model?
23. Write the loss function of regularized logistic regression model?
24. Define regularization parameter $\lambda$. What happend if you use very high or very low $\lambda$?
25. How can we find out optimal value of $\lambda$?
26. Define $L1$ and $L2$ regularization?
27. Explain pros and cons of $L1$ and $L2$ regularization?
28. Write pseudocode of implementing $L2$ regularization in a neural network.
29. Explain how does the implementation of $L2$ regularization in NN leads to weight decay situation?
30. Why regularization reduces overfitting?
31. List down some widely use regularization techniques for NN.
32. How does dropout technique helps in reducing overfitting situations?
33. Implement dropout using inverted dropout technique.
34. Why does drop-out work?
35. What are some downside of using dropout as a regularization technique?
36. Explain what does **keep-prob** value depicts?
37. Does droput has any ensembling effect?
38. Define data augmentation. How does it helps in preventing overfitting?
39. What are different data augmentation methods for image data?
40. What is early stopping and how does it helps in preventing overfitting?
41. List down downsides of using early stopping?
42. What is the intuition behind early stopping?
43. How do we decide when to stop in early stopping method? Illustrate it using a diagram.
44. Why should we normalize the input features to neural networks?
45. How does normalizing input features speed training process?
46. Write pseudocode for normalizing input features?
47. What is the difference between normalization and standardization?
48. Comment on shape of the cost function in case of normalized and non-normalized inputs?
49. What is the issue with training very deep neural networks?
50. Explain vanishing and exploding gradient issue with DNN?
51. What is the impact of weight intialization on vanishing/exploding gradient issues?
52. What are differnt strategies of weight initilization?
53. How does the Xavier and Xe initilization techniques work? Explain the math behind them?
54. Suppose you are implementing a back propagation algorithm for you neural network model, How can you check if the algorithm is working as intended?
55. Write steps for gradient checking in NN.
56. During gradient check why it is advisable not to use dropout?
57. What is gradient descent?
58. Define batch and mini-batch gradient descent.
59. Compare and contrast between batch and mini-batch gradient descent.
60. Draw the $cost$ vs $\\# iterations$ in case of batch gradient descent and mini-batch gradient descent. Explain the observations.
61. Write the pseduocode of calculations involve in mini-batch gradient descent.
62. When to use batch GD over mini-batch GD?
63. How to choose batch size in mini-batch gradient descent?
64. What are typical mini-batches sizes that are widely used?
65. What are various techniques to speed gradient descent algorithm?
66. What do you mean by Exponential Weighted Average? Explain its application.
67. Write the equation involves in EWA?
68. What is bias correction in exponentially weighted averages?
69. How bias corrections can be helpful?
70. What is the idea behind gradient descent with momentum?
71. Illsutrate the effect of using momentum with gradient descnet on training rate.
72. Implement gradient descent with momentum and also mention the hyper-parameters involve in it.
73. Explain the intuition behind the less number of oscillations in case of gradient descent with momentum.
74.  

## Structuring Machine Learning Projects

## Convolutional Neural Networks

## Sequence Models
