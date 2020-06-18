# Name That Dog
Applying transfer learning to Resnet18 model to classify dog breeds.

## PyTorch

### Autograd: Automatic Differentiation
Central to all neural networks in PyTorch is the autograd package. Let’s first briefly visit this, and we will then go to training our first neural network.

The autograd package provides automatic differentiation for all operations on Tensors. It is a define-by-run framework, which means that your backprop is defined by how your code is run, and that every single iteration can be different.

Let us see this in more simple terms with some examples.

### Tensor
`torch.Tensor` is the central class of the package. If you set its attribute `.requires_grad` as True, it starts to track all operations on it. When you finish your computation you can call `.backward()` and have all the gradients computed automatically. The gradient for this tensor will be accumulated into .grad attribute.

To stop a tensor from tracking history, you can call `.detach()` to detach it from the computation history, and to prevent future computation from being tracked.

To prevent tracking history (and using memory), you can also wrap the code block in with `torch.no_grad():`. This can be particularly helpful when evaluating a model because the model may have trainable parameters with requires_grad=True, but for which we don't need the gradients.

There’s one more class which is very important for autograd implementation - a `Function`.

`Tensor` and `Function` are interconnected and build up an acyclic graph, that encodes a complete history of computation. Each tensor has a `.grad_fn` attribute that references a `Function` that has created the `Tensor` (except for Tensors created by the user - their `grad_fn` is None).

If you want to compute the derivatives, you can call `.backward()` on a `Tensor`. If `Tensor` is a scalar (i.e. it holds a one element data), you don’t need to specify any arguments to `backward()`, however if it has more elements, you need to specify a gradient argument that is a tensor of matching shape.

### Neural Nets
Neural networks can be constructed using the `torch.nn` package.

Now that you had a glimpse of autograd, nn depends on autograd to define models and differentiate them. An `nn.Module` contains layers, and a method forward(input)that returns the output.


A typical training procedure for a neural network is as follows:

1. Define the neural network that has some learnable parameters (or weights)

2. Iterate over a dataset of inputs

3. Process input through the network

4. Compute the loss (how far is the output from being correct)

5. Propagate gradients back into the network’s parameters

6. Update the weights of the network, typically using a simple update rule: `weight = weight - learning_rate * gradient`

### Notes
* `torch.Tensor` - A multi-dimensional array with support for autograd operations like `backward()`. Also holds the gradient w.r.t. the tensor.
* `nn.Module` - Neural network module. Convenient way of encapsulating parameters, with helpers for moving them to GPU, exporting, loading, etc.
* `nn.Parameter `- A kind of Tensor, that is automatically registered as a parameter when assigned as an attribute to a Module.
* `autograd.Function` - Implements forward and backward definitions of an autograd operation. Every Tensor operation creates at least a single Function node that connects to functions that created a Tensor and encodes its history.
* **`torch.nn` only supports mini-batches**. The entire torch.nn package only supports inputs that are a mini-batch of samples, and not a single sample.
* For example, `nn.Conv2d` will take in a 4D Tensor of `nSamples x nChannels x Height x Width`.
* If you have a single sample, just use `input.unsqueeze(0)` to add a fake batch dimension.


## Loss Functions

### Mean Absolute Error
It measures the numerical distance between the estimated and actual value. It is the simplest form of error metric. The absolute value of the error is taken because if we don’t then negatives will cancel out the positives. This isn’t useful to us, rather it makes it more unreliable.
The lower the value of MAE, better is the model. We can not expect its value to be zero, because it might not be practically useful. This leads to wastage of resources. For example, if our model’s loss is within 5% then it is alright in practice, and making it more precise may not really be useful.

When to use it?
* **Regression** problems
* Simplistic model
* As neural networks are usually used for complex problems, this function is rarely used.

### Mean Square Error Loss
The squaring of the difference of prediction and actual value means that we’re amplifying large losses. If the classifier is off by 200, the error is 40000 and if the classifier is off by 0.1, the error is 0.01. This penalizes the model when it makes large mistakes and incentivizes small errors.

When to use it?
* **Regression** problems.
* The numerical value features are not large.
* Problem is not very high dimensional.

### Smooth L1 Loss

It uses a squared term if the absolute error falls below 1 and an absolute term otherwise. It is less sensitive to outliers than the mean square error loss and in some cases prevents exploding gradients. In mean square error loss, we square the difference which results in a number which is much larger than the original number. These high values result in exploding gradients. This is avoided here as for numbers greater than 1, the numbers are not squared.

When to use it?
* **Regression**.
* When the features have large values.
* Well suited for most problems.

### Negative Log-Likelihood Loss
It maximizes the overall probability of the data. It penalizes the model when it predicts the correct class with smaller probabilities and incentivizes when the prediction is made with higher probability. The logrithm does the penalizing part here. Smaller the probabilities, higher will be its logrithm. The negative sign is used here because the probabilities lie in the range [0, 1] and the logrithms of values in this range is negative. So it makes the loss value to be positive.

When to use it?
* **Classification**.
* Smaller quicker training.
* Simple tasks.

### Cross-Entropy Loss
Cross-entropy as a loss function is used to learn the probability distribution of the data. While other loss functions like squared loss penalize wrong predictions, cross entropy gives a greater penalty when incorrect predictions are predicted with high confidence. What differentiates it with negative log loss is that cross entropy also penalizes wrong but confident predictions and correct but less confident predictions, while negative log loss does not penalize according to the confidence of predictions.

When to use it?
* **Classification** tasks
* For making confident model i.e. model will not only predict accurately, but it will also do so with higher probability.
* For higher precision/recall values.

### Kullback-Leibler divergence
It is quite similar to cross entropy loss. The distinction is the difference between predicted and actual probability. This adds data about information loss in the model training. The farther away the predicted probability distribution is from the true probability distribution, greater is the loss. It does not penalize the model based on the confidence of prediction, as in cross entropy loss, but how different is the prediction from ground truth. It usually outperforms mean square error, especially when data is not normally distributed. The reason why cross entropy is more widely used is that it can be broken down as a function of cross entropy. Minimizing the cross-entropy is the same as minimizing KL divergence.
KL = — xlog(y/x) = xlog(x) — xlog(y) = Entropy — Cross-entropy

When to use it?
* **Classification**
* Same can be achieved with cross entropy with lesser computation, so avoid it.

## Acknowledgements
Dog breed JSON from:
https://github.com/dariusk/corpora/blob/master/data/animals/dogs.json

Alano-Español
Alaskan-Klee-Kai
