# Effective Batch Size and LR

As you scale up the number of nodes, the effective batch size (the amount of items used for model updates) increases as well:

```
effective_batch_size = batch_size * world_size
```

As you may know, increasing the batch size means that the variance of the data that your model is training on decreases, meaning your gradients will be much smoother. This directly impacts the dynamics of how your model learns and changes!

If you want to **exactly match the dynamics of single gpu training** when moving to multi node training, this chapter is aimed at you!

## Scaling Rules

If you want exact training dynamics, you have to also scale the learning rate. However, this depends on what optimizer you are using. The exact rules are not fully understood, and you can look into the following papers for more information:

- [Exploring Learning Rate Scaling Rules for Distributed ML Training on Transient Resources](https://anakli.inf.ethz.ch/papers/learning_rate_distribml22.pdf)

As of writing this, the most common rules that people use to scale learning rate are:

### Linear scaling rule

```python
lr = args.lr * dist.get_world_size()
```

This was first reported in the large minibatch SGD paper above. However this doesn't quite produce exactly the same training dynamics, and the paper actually used a **factor of the world size**.

NOTE: **Be careful when using this for optimizers other than SGD**

References:
- [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/pdf/1706.02677)

### Square root scaling rule

```python
lr = args.lr * numpy.sqrt(dist.get_world_size())
```

This is proposed for use with the Adam optimizer, and maintains the square root of the variance of the gradient when scaling the number of batches.

References:
- [One weird trick for parallelizing convolutional neural networks](https://arxiv.org/pdf/1404.5997)
- [Large-Batch Training for LSTM and Beyond](https://arxiv.org/pdf/1901.08256)
