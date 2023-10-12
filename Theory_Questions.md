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
74. What is RMSProp algorithm and how it works?
75. Explain the maths behind RMSProp algorithm?
76. What are the hyper-parameters involve in RMSProp?
77. Define Adam optimization algorithm.
78. What does Adam stands for?
79. How does Adam optimization combine momentum and RMSProp together?
80. What are the hyper-parameters involve in Adam algorithm? Comment on their common values.
81. Write down the equations involve in Adam optimization algorithm.
82. What is learning rate decay and why it is used in speeding up training algorithm?
83. Explain the intuition behind decay in learning rate.
84. Write equations and hyper-parameters involve in learning rate decay?
85. What are saddle points of a function?
86. How can we overcome the local optima problems?
87. How does plateus of a cost functions create problem in training?
88. What are different ways of tuning hyper-parameters?
89. What is Pandas vs Caviar strategy in finding optimal sets of hyperparamters?
90. Define batch normalization.
91. What are benefits of using batch normalization?
92. Write down setps and equations involve in batch norm.
93. Should we normalize the value before the activation function, or whether you should normalize the value after applying the activation function?
94. Why does batch norm works? Explain the intuition behind it.
95. Does batch norm has any regularization effect?
96. How does batch norm work during training and test time?
97. Write down the equations used during test time.
98. What should be the activation function in case of multi class classification?
99. How softmax function works?
100. Write the expression of softmax function.

## Structuring Machine Learning Projects

## Convolutional Neural Networks

1. List down some example of computer vision tasks.
2. Define convolution mathematically.
3. Write the expression of convolutional integretion.
4. Evaluate the below expression where `*` represents convolution operator.
   ```plaintext
   10  10  10  0  0  0       
   10  10  10  0  0  0       1  0  -1
   10  10  10  0  0  0    *  1  0  -1
   10  10  10  0  0  0       1  0  -1
   10  10  10  0  0  0
   10  10  10  0  0  0
   ```
5. Define what do you mean by edges in an image?
6. Ilustrate how can you use convolution to find edges in an image?
7. You are given a filter for verticle edge detection, find out corresponding horizontal edge detector.
   ```plaintext
   1 0 -1
   1 0 -1
   1 0 -1
   ```
8. Write down the expression of following
   1. Sobel filter
   2. Scharr filter
9. How do we can find out the appropriate parameters of edge detector filters?
10. What will be the resultant diameter of a $n\*n$ image convolved with $f\*f$ filter?
11. What is padding? Why do we need to use them in building deep neural networks for image tasks?
12. Suppose you have a $n\*n$ image and $p$ is the padding amount, you are using a convolution filter of dimension $f\*f$. What will the resultant dimension?
13. Based on amount of padding, what are different types of convolutions?
14. Distinguise between valid and same convolutions.
15. What is the reason behind using square and odd length of convolution filters?
16. What is strided convolution? Explain it with an example?
17. Suppose you have an $N \* N$ image, they convolve with an $F \* F$ filter, and if you use padding P and stride S. Comment on final output dimension.
18. Suppose $n_c$ is number of channels in your image, what should be $n_c$ for filter?
19. How do we operate convolution over volume? Illustrate.
20. If you have 10 filters that are $3\*3\*3$ in one layer of a neural network, how many parameters does that layer have?
21. What is the issue with feed forward neural netowrok while dealing with image related tasks?
22. What are the benefits of using CNNs instead of FFNNs?
23. Suppose for layer $l$ of a convolution layer, you have following info
    1. $f^{[l]}$ = filter size
    2. $p^{[l]}$ = padding
    3. $s^{[l]}$ = stride
    4. $n_c^{[l]}$ = number of filters
   What will be the dimensions of following:
     1. Each filter
     2. Activations
     3. Weights
     4. Bias
     5. Input matrix
     6. Output matrix
24. How many layers a typical ConvNet contains? Write their Names.
25. What is a pooling layer? Why we need them in ConvNets?
26. Where do we use pooling layers in ConvNets?
27. What are the different types of pooling layers?
28. How does the max pooling layer work?
29. What are the intuition behind using max pooling layers?
30. Name the parameters and hyper-parameters involve in pooling layers.
31. Draw the architecture of following Networks.
    1. LeNet-5
    2. AlexNet
    3. VGG-16
32. How many parameters does the following networks had?
    1. LeNet-5
    2. AlexNet
    3. VGG-16
33. What are ResNets? Explain the building block behind it.
34. How does ResNets overcome with vanishing gradient problem?
35. Write down the equations in envole in a residual block.
36. What are the skip conections in ResNet and how does it helps in training deep NN models?
37. 


## Sequence Models
1. Give some examples of sequence data.
2. What are different ways of representing words in NLP?
3. What are recurrent neural networks?
4. What are the issues with fully connected NN while using for sequence data?
5. Draw schematic diagram of a rnn block?
6. For forward propagation in rnn, write the equations involve a given layer.
7. How does backpropagation works through time in RNNs?
8. What are different types of RNNs based on its input output architecture?
9. WHat are some exaples of language models?
10. What are some pros and cons of using character level language model over word based language model?
11. What are some drawbacks of using vanilla RNNs for sequence modeling?
12. Why does Vanishing gradient issue happens with vanilla RNNs?
13. How can you address exploding gradient issues in RNNs?
14. What are the remedies for vanishing gradient issues in RNNs?
15. What are some popular RNNs design?
16. What are GRUs? How do they overcome with vanishing gradient problem?
17. What are different components of a GRU unit? Write down the equations for each gates?
18. Draw a LSTM unit and anotate all of its important component.
19. State the equations governing each gates of a LSTM cell.
20. What are bidirectional RNNs? How do we train them?
21. What is benefit of using bidirectional rnn over unidirectional ones?
22. State one limitation of bidirectional rnn?
23.  
