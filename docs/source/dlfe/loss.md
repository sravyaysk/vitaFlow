## Loss/Cost Functions
Loss Function: For a given input, we’d like to train our model to maximize the probability 
it assigns to the right answer. To do this, we’ll need to efficiently compute the 
conditional probability `$p(Y | X)$`. The function `$p(Y \mid X)$`  should also be 
differentiable, so we can use gradient descent.

### Classification
#### Cross Entropy
Activation Function: Softmax

### Regression
#### Squarred Error
Activation Function : Tanh, Sigmoid

### Sequence
#### CRF