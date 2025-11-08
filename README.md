# Distributed Training Guide

🔥 News 🔥

- [Pytorch Conference 2025 slides](https://docs.google.com/presentation/d/126BQcCJti_z3NQ7cGbm2iMBDVG7-0ekitxU3pvJ_WQw/edit?usp=sharing)
- [Neurips 2024 slides](https://docs.google.com/presentation/d/1ANMmkOGaruYKTvhnsAbZgI9GrdMliNvibWGuNYw6HX8/edit?usp=sharing)

<img src="https://lambdalabs.com/hubfs/distriubuted-training-guide.png" width="400px" />

Ever wondered how to train a large neural network across a giant cluster? Look no further!

This is a comprehensive guide on best practices for distributed training, diagnosing errors, and fully utilizing all resources available. It is organized into sequential chapters, each with a `README.md` and a `train_llm.py` script in them. The readme will discuss both the high level concepts of distributed training, and the code changes introduced in that chapter.

The guide is written entirely in very minimal standard pytorch, using `transformers` and `datasets` for models and data, respectively. No other library is used for distributed code - the distributed stuff is entirely in pytorch.

1. [Chapter 1](./01-single-gpu/) - A standard Causal LLM training script that runs on a **single GPU**.
2. [Chapter 2](./02-distributed-data-parallel/) - Upgrades the training script to support **multiple GPUs and to use DDP**.
3. [Chapter 3](./03-job-launchers/) - Covers how to **launch training jobs** across clusters with multiple nodes.
4. [Chapter 4](./04-fully-sharded-data-parallel/) - Upgrades the training script to **use FSDP** instead of DDP for more optimal memory usage.
5. [Chapter 5](./05-training-llama-405b/) - Upgrades the training script to **train Llama-405b**.
6. [Chapter 6](./06-tensor-parallel/) - Upgrades our single GPU training script to support **tensor parallelism**.
7. [Chapter 7](./07-2d-parallel/) - Upgrades our TP training script to use **2d parallelism (FSDP + TP)**.
8. [Alternative Frameworks](./alternative-frameworks/) - Covers different frameworks that all work with pytorch under the hood.
9. [Diagnosing Errors](./diagnosing-errors/) - Best practices and how tos for **quickly diagnosing errors** in your cluster.
10. [Related Topics](./related-topics/) - Topics that you should be aware of when distributed training.


Questions this guide answers:

- How do I update a single gpu training/fine tuning script to run on multiple GPUs or multiple nodes?
- How do I diagnose hanging/errors that happen during training?
- My model/optimizer is too big for a single gpu - how do I train/fine tune it on my cluster?
- How do I schedule/launch training on a cluster?
- How do I scale my hyperparameters when increasing the number of workers?

Best practices for logging stdout/stderr and wandb are also included, as logging is vitally important in diagnosing/debugging training runs on a cluster.

Each of the training scripts is aimed at training a causal language model (i.e. gpt/llama).

## Set up

### Clone this repo

```bash
git clone https://github.com/LambdaLabsML/distributed-training-guide.git
```

### Virtual Environment

```bash
cd distributed-training-guide
python3 -m venv venv
source venv/bin/activate
python -m pip install -U pip
pip install -U setuptools wheel
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

### wandb

This tutorial uses `wandb` as an experiment tracker.

```bash
wandb login
```

<p align="center">
🦄 Other exciting ML projects at Lambda: <a href="https://news.lambda.ai/news/today">ML Times</a>, <a href="https://lambdalabsml.github.io/Open-Sora/introduction/">Text2Video</a>, <a href="https://lambdalabs.com/gpu-benchmarks">GPU Benchmark</a>.
</p>
