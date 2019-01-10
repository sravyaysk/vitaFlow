# Introduction

A typical deep learning project starts with preparing the dataset (**Data**), preprocessing the data such that the domain data is 
is getting transformed to model specific numeric data (**Data Iterators**). Then comes the model building part by stacking 
the neural network layers (with Activation Functions). Followed by defining a **Cost/Loss Function** which compares the 
ground truth value with predicted value and gives out a numerical value/tensor. Once the loss function is established, 
then the natural following step would be considering the the loss value/tesnor and use a **Optimization Function** which 
uses some special algorithms along with back propagation to bring down the loss value/tensor  by adjusting the network weights.

**Epoch** : One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE.
       
**Batch Size** : Total number of training examples present in a single batch.
    
**Iterations/Steps** : Iterations is the number of batches needed to complete one epoch.    

```
Eg: We can divide the dataset of 2000 examples into batches of 500 then it will take 4 iterations to complete 1 epoch.

Where: 
Epoch is 1
Batch Size is 500
Iterations is 4
```

## Must Read Blogs
- [https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b)
