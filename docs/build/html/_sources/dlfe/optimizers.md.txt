## Optimization Algorithms

One of the best material on the topic can be found here @ http://ruder.io/optimizing-gradient-descent/

### Batch gradient descent
Vanilla gradient descent, aka batch gradient descent, computes the gradient of the cost function w.r.t. to the parameters θ
for the entire training dataset:

$$\theta = \theta - \eta \cdot \nabla_\theta J( \theta)$$

### Stochastic gradient descent
Stochastic gradient descent (SGD) in contrast performs a parameter update for each training example 
`$x^(i)$` and label `$y^(i)$`:

$$\theta = \theta - \eta \cdot \nabla_\theta J( \theta; x^{(i)}; y^{(i)})$$

### Mini-batch gradient descent
Mini-batch gradient descent finally takes the best of both worlds and performs an update for every mini-batch of 
n training examples:

$$\theta = \theta - \eta \cdot \nabla_\theta J( \theta; x^{(i:i+n)}; y^{(i:i+n)})$$

### Momentum
Momentum is a method that helps accelerate SGD in the relevant direction and dampens oscillations. 
It does this by adding a fraction γ of the update vector of the past time step to the current update vector:

$$
\begin{align} 
\begin{split} 
v_t &= \gamma v_{t-1} + \eta \nabla_\theta J( \theta) \newline
\theta &= \theta - v_t 
\end{split} 
\end{align}
$$

### Adagrad

- [Paper](http://jmlr.org/papers/v12/duchi11a.html)
- [TF API](https://www.tensorflow.org/api_docs/python/tf/train/AdagradOptimizer)

Adagrad is an algorithm for gradient-based optimization that does just this: It adapts the learning rate to the parameters, 
performing smaller updates (i.e. low learning rates) for parameters associated with frequently 
occurring features, and larger updates (i.e. high learning rates) for parameters associated with 
infrequent features. For this reason, it is well-suited for dealing with sparse data.

### Adadelta

- [TF API](https://www.tensorflow.org/api_docs/python/tf/train/AdadeltaOptimizer)

Adadelta is an extension of Adagrad that seeks to reduce its aggressive, monotonically decreasing 
learning rate. Instead of accumulating all past squared gradients, Adadelta restricts the window of 
accumulated past gradients to some fixed size w.

### Adam

- [TF API](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer)    

Adaptive Moment Estimation (Adam) is another method that computes adaptive learning rates for each parameter. 
In addition to storing an exponentially decaying average of past squared gradients `$v_t$`
like Adadelta, Adam also keeps an exponentially decaying average of past gradients `$m_t$`, similar to momentum. 
Whereas momentum can be seen as a ball running down a slope, Adam behaves like a heavy ball with friction, 
which thus prefers flat minima in the error surface. We compute the decaying averages of past and past squared gradients `$m_t$` and `$v_t$` respectively as follows:

$$
\begin{align} 
\begin{split} 
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \newline
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \newline
\end{split} 
\end{align}
$$

$$
\begin{align} 
\begin{split} 
\hat{m}_t &= \dfrac{m_t}{1 - \beta^t_1} \newline
\hat{v}_t &= \dfrac{v_t}{1 - \beta^t_2} \newline
\end{split} 
\end{align}
$$

$$
\theta_{t+1} = \theta_{t} - \dfrac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
$$
