# Training a 405B model with 2d Parallelism

So far we've just been using data parallel techniques. You may have heard of other parallelism techniques, and indeed the [Llama 405B paper](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/) actually uses 4D parallelism when training the 405B model:

1. Data parallel (FSDP as we've learned)
2. Tensor parallel (this chapter)
3. Context parallel (For long context lengths)
4. Pipeline/model parallel

In this chapter we are going to be adding tensor parallelism to our 405b training script from [chapter 6](../06-training-llama-405b/) to see what improvements we get.

## Basics: What is tensor parallelism?

Some operations can be easily split among multiple workers without having to share any parameters. If you remember with fully sharded data parallel, we still have to gather all the weights onto each node. That is due to the way the operations we were wrapping worked.

Tensor Parallelism is most often applied with Linear layers, which occur a lot in Transformer architecture. Basically the weight & bias matrix in the Linear layer are split up across your workers, and then when applying the forward to the tensor, only the input tensor needs to be split across the workers, not the weights.

This ends up being much cheaper/faster!

## Useful References

For completeness here are the relevant docs/guides from pytorch on how to achieve this:
- [TP API docs](https://pytorch.org/docs/stable/distributed.tensor.parallel.html#tensor-parallelism-torch-distributed-tensor-parallel)
- [2d Parallelism Tutorial](https://pytorch.org/tutorials/intermediate/TP_tutorial.html#large-scale-transformer-model-training-with-tensor-parallel-tp)
- [Device Mesh tutorial](https://pytorch.org/tutorials/recipes/distributed_device_mesh.html)
- [PyTorch Lightning TP Tutorial](https://lightning.ai/lightning-ai/studios/tensor-parallelism-supercharging-large-model-training-with-pytorch-lightning)

## The Tensor Parallel API

Here we are going to give a brief explanation of how the api we are going to be using works.

- [tp.RowwiseParallel()]() shards the module's weights in a row wise fashion.
    - Inputs by default are sharded on last dimension
    - Outputs by default are replicated on all workers
- [tp.ColwiseParallel()]() shards the module's weights in a col wise fashion.
    - Inputs by default are replicated on all workers
    - Outputs by default are sharded on last dimension
- [tp.SequenceParallel()]() shards the input/output across dimension 1. Module weights are NOT sharded.
- [tp.loss_parallel()]() shards CrossEntropyLoss computation. **Requrires model output to be sharded on class dimension**
- [tp.PrepareModuleInput()]() let's you change the sharding configuration of input tensors
- [torch.distributed._tensor.Shard(dim=X)]() indicates a tensor should be sharded along dimension X
- [torch.distributed._tensor.Replicate()]() indicates a tensor should be replicated among all workers.

How all of these things interact is actually very subtle and complex, which is why this guide is useful!

You can also change most of the default behavior with arguments to these classes. For example, you can change RowwiseParallel to assume the input is replicated instead of sharded.

### Colwise Linear

TODO

![image](https://storage.googleapis.com/lightning-avatars/litpages/01hyz8vg94nc6nk7t10rt8jpt1/a2fe38f3-4a73-4b0f-80da-c273d14cadd9.jpeg)

Source: [PyTorchLightning](https://lightning.ai/lightning-ai/studios/tensor-parallelism-supercharging-large-model-training-with-pytorch-lightning#column-wise-parallel)

### Rowwise Linear

TODO

![image](https://storage.googleapis.com/lightning-avatars/litpages/01hyz8vg94nc6nk7t10rt8jpt1/6b715900-897d-4b3d-a1b6-8ce48f213acf.jpeg)

Source: [PyTorchLightning](https://lightning.ai/lightning-ai/studios/tensor-parallelism-supercharging-large-model-training-with-pytorch-lightning#row-wise-parallel)


### Chaining Linears

TODO

![image](https://storage.googleapis.com/lightning-avatars/litpages/01hyz8vg94nc6nk7t10rt8jpt1/a1bc6e8a-7146-44c6-b6cf-eec124cfbf74.jpeg)

Source: [PyTorchLightning](https://lightning.ai/lightning-ai/studios/tensor-parallelism-supercharging-large-model-training-with-pytorch-lightning#combined-parallel-layers)

### Parallelizing Embedding with RowwiseParallel

TODO

### Parallelizing Norm Layers with SequenceParallel

TODO

### Parallelizing Cross Entropy Loss

TODO

## Implementation

There are a couple of parts to implementing this. For starters we are going to need what pytorch calls a [Device Mesh](https://pytorch.org/docs/stable/distributed.html#torch.distributed.device_mesh.DeviceMesh). Why? Because this will let us split our GPUs into groups based on what node they are on.

For a high level view first, we are going to:

1. Tensor parallelism in each node (so split Linear layers up across the node)
2. Data parallelism (FSDP) across nodes

A DeviceMesh helps us achieve this.

```python
# NOTE: assumes all nodes have the same number of gpus
gpus_on_node = torch.cuda.device_count()
num_nodes = dist.get_world_size() // gpus_on_node
mesh = dist.device_mesh.init_device_mesh(
    "cuda",
    (num_nodes, gpus_on_node),
    mesh_dim_names=("dp", "tp"),
)
```

### Model Architecture Reference

For reference for the next couple of sections, here is the architecture for 405B:

```python
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 16384)
    (layers): ModuleList(
      (0-125): 126 x LlamaDecoderLayer(
        (self_attn): LlamaSdpaAttention(
          (q_proj): Linear(in_features=16384, out_features=16384, bias=False)
          (k_proj): Linear(in_features=16384, out_features=1024, bias=False)
          (v_proj): Linear(in_features=16384, out_features=1024, bias=False)
          (o_proj): Linear(in_features=16384, out_features=16384, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=16384, out_features=53248, bias=False)
          (up_proj): Linear(in_features=16384, out_features=53248, bias=False)
          (down_proj): Linear(in_features=53248, out_features=16384, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((16384,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((16384,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((16384,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=16384, out_features=128256, bias=False)
)
```

### Applying tensor parallelism to our model

Note that this is done *before* passing our model to FSDP. Luckily this all works very seamlessly with meta models.

```python
import torch.distributed.tensor.parallel as tp
```
TODO


### Using our mesh with FSDP

For this, we can just pass our dp mesh directly to the FSDP constructor:

```python
model = FSDP(..., device_mesh=mesh["dp"])
```

Note that this allows us to try other versions of sharding.

You can also just pass the entire mesh into device_mesh. TODO what should be done??

Here are three options:
1. `sharding_strategy=ShardingStrategy.FULL_SHARD` and `device_mesh=mesh["dp"]`

TODO

2. `sharding_strategy=ShardingStrategy.SHARD_GRAD_OP` and `device_mesh=mesh`

TODO

3. `sharding_strategy=ShardingStrategy.HYBRID_SHARD` and `device_mesh=mesh`

TODO

4. `sharding_strategy=ShardingStrategy._HYBRID_SHARD_ZERO2` and `device_mesh=mesh`

TODO

### TP Groups Must have Identical inputs!!!

TODO????

