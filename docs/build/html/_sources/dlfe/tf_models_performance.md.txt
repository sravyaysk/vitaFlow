# TF Model Performance

## Vanishing Gradient 

## [Batch Normalization](https://arxiv.org/abs/1502.03167)

- Preventing the early saturation of non-linear activation functions like the sigmoid function, assuring that all 
input data is in the same range of values, etc. 
- **Internal covariate shift** : Weight distributions due to activation function in hidden layer constantly changes during the training
cycle. This slows down the training process because each layer must learn to adapt themselves to a new distribution 
in every training step.
- Calculate the mean and variance of the layers input. Where $m$ is number of samples in the current batch.
$$
Batch\ mean : \mu_B = \frac{1}{m}\sum_{i=1}^m x_i \\
Batch\ Variance :  \sigma_B^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_B)^2 \\
$$

- Normalize the layer inputs using the previously calculated batch statistics.
$$
\hat{x_i} = \frac{x_i-\mu_B}{\sqrt{\sigma^2_B + \epsilon}}
$$
-  Scale and shift in order to obtain the output of the layer.
$$
y_i = \gamma \hat{x_i} + \beta
$$
- **γ** and **β** are learned during training along with the original parameters of the network.
- During test (or inference) time, the mean and the variance are fixed. They are calculated using the previously calculated means and variances of each training batch.
- So, if each batch had m samples and there where j batches:
$$
Inference mean : E_x = \frac{1}{m}\sum_{i=1}^j\mu_B^{i} \\
Inference Variance : Var_x = (\frac{m}{m-1})\frac{1}{m}\sum_{i=1}^j\sigma_B^{2i} \\
Inference scaling/shifting : y = x\frac{\gamma}{\sqrt{Var_x + \epsilon}}+\beta\frac{\gamma E_x}{\sqrt{Var_x + \epsilon}}
$$

### API
- [https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization](https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization)
