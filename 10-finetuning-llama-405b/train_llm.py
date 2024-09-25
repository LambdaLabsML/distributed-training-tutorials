import argparse
from contextlib import contextmanager
from datetime import timedelta
import functools
from itertools import chain
import json
import multiprocessing
import random
import time
from pathlib import Path
import logging

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch import distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    offload_wrapper,
)
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel,
    BackwardPrefetch,
    CPUOffload,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
    set_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint.state_dict_loader import load
from torch.distributed.checkpoint.state_dict_saver import save


import numpy
import wandb
import tqdm
import datasets
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
)

_LOGGER = logging.getLogger(__name__)


@record
def main():
    parser = _get_parser()
    args = parser.parse_args()

    dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "mpi")

    rank = dist.get_rank()
    local_rank = rank % torch.cuda.device_count()
    world_size = dist.get_world_size()

    logging.basicConfig(
        format=f"[rank={rank}] [%(asctime)s] %(levelname)s:%(message)s",
        level=logging.INFO,
    )

    _LOGGER.info(args)
    _LOGGER.info(f"local_rank={local_rank} rank={rank} world size={world_size}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    numpy.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.bfloat16
    torch.cuda.set_device(device)

    config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        # NOTE: only load the weights on rank 0
        #       these will be sent to other ranks
        #       with `sync_module_states=True` later
        device_map="cpu" if rank == 0 else "meta",
        attn_implementation="flash_attention_2",
    )

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    _LOGGER.info(
        f"Before FSDP: {torch.cuda.memory_stats(device)['allocated_bytes.all.current'] * 1e-9}gb allocated"
    )

    from transformers.models.llama.modeling_llama import LlamaDecoderLayer

    wrap_policy = functools.partial(
        transformer_auto_wrap_policy, transformer_layer_cls={LlamaDecoderLayer}
    )
    model = FullyShardedDataParallel(
        model,
        device_id=local_rank,
        param_init_fn=lambda m: m.to_empty(device=device, recurse=False),
        sync_module_states=True,
        # NOTE: FULL_SHARD is equivalent to deepspeed ZeRO stage 3
        auto_wrap_policy=wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        cpu_offload=CPUOffload(offload_params=args.cpu_offload == "on"),
        backward_prefetch=getattr(BackwardPrefetch, args.bwd_prefetch, None),
    )

    _LOGGER.info(
        f"After FSDP: {torch.cuda.memory_stats(device)['allocated_bytes.all.current'] * 1e-9}gb allocated"
    )
    _LOGGER.info(f"FSDP architecture: {model}")

    wrapper_fn = {
        "checkpoint": checkpoint_wrapper,
        "offload": offload_wrapper,
        "in-memory": None,
    }[args.activations]
    if wrapper_fn is not None:
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=wrapper_fn,
            check_fn=lambda l: "Attention" in l.__class__.__name__,
        )

    # NOTE: since this can download data, make sure to do the main process first
    # NOTE: This assumes that the data is on a **shared** network drive, accessible to all processes
    with rank0_first():
        train_data = _load_and_preprocess_data(args, tokenizer, config)
    _LOGGER.info(f"{len(train_data)} training samples")

    dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        collate_fn=default_data_collator,
        num_workers=1,
        prefetch_factor=2,
        # NOTE: this sampler will split dataset evenly across workers
        sampler=DistributedSampler(train_data, shuffle=True, drop_last=True),
    )
    _LOGGER.info(f"{len(dataloader)} batches per epoch")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=1000, eta_min=args.lr * 1e-2
    )

    exp_dir: Path = Path(args.save_dir) / args.experiment_name

    # NOTE: full_state_dict=False means we will be saving sharded checkpoints.
    ckpt_opts = StateDictOptions(full_state_dict=False, cpu_offload=True)

    # attempt resume
    state = {
        "epoch": 0,
        "global_step": 0,
        "epoch_step": 0,
        "running_loss": 0,
    }
    resumed = False
    if (exp_dir / "state.json").exists():
        sharded_model_state, sharded_optimizer_state = get_state_dict(
            model, optimizer, options=ckpt_opts
        )
        load(
            dict(model=sharded_model_state, optimizer=sharded_optimizer_state),
            checkpoint_id=exp_dir / "checkpoint",
        )
        set_state_dict(
            model,
            optimizer,
            model_state_dict=sharded_model_state,
            optim_state_dict=sharded_optimizer_state,
            options=ckpt_opts,
        )
        lr_scheduler.load_state_dict(
            torch.load(
                exp_dir / "lr_scheduler.pt", map_location=device, weights_only=True
            )
        )
        with open(exp_dir / "state.json") as fp:
            state = json.load(fp)
        resumed = True
    _LOGGER.info(f"Resumed={resumed} | {state}")

    dist.barrier()
    if rank == 0:
        # NOTE: assuming directory is shared across all nodes, that's why we do rank instead of local_rank
        _LOGGER.info(f"Creating experiment root directory")
        exp_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    (exp_dir / f"rank-{rank}").mkdir(parents=True, exist_ok=True)
    _LOGGER.info(f"Worker saving to {exp_dir / f'rank-{rank}'}")

    if rank == 0:
        wandb.init(
            project="distributed-training-guide",
            dir=exp_dir,
            name=f"{args.experiment_name}",
            id=f"{args.experiment_name}",
            resume="must" if resumed else None,
            save_code=True,
            config={
                "args": vars(args),
                "embedding_size": len(tokenizer),
                "training_data_size": len(train_data),
                "num_batches": len(dataloader),
                "world_size": world_size,
            },
        )

    timers = {
        k: LocalTimer(device)
        for k in ["data", "forward", "backward", "update", "waiting"]
    }

    for state["epoch"] in range(state["epoch"], args.num_epochs):
        _LOGGER.info(f"Begin epoch {state['epoch']} at step {state['epoch_step']}")

        progress_bar = tqdm.tqdm(range(len(dataloader)))
        if state["epoch_step"] > 0:
            progress_bar.update(state["epoch_step"])

        batches = iter(dataloader)

        for i_step in range(len(dataloader)):
            with timers["data"], torch.no_grad():
                batch = next(batches)
                batch = {k: v.to(device=device) for k, v in batch.items()}

            if i_step < state["epoch_step"]:
                # NOTE: for resuming
                continue

            _LOGGER.info(f"{rank=} {i_step=} data sent to device")

            with timers["waiting"]:
                dist.barrier()

            with timers["forward"]:
                outputs = model(**batch)

            _LOGGER.info(f"{rank=} {i_step=} forward() finished")

            with timers["waiting"]:
                dist.barrier()

            with timers["backward"]:
                optimizer.zero_grad()
                _LOGGER.info(f"{rank=} {i_step=} optimizer.zero_grad() finished")

                outputs.loss.backward()
                _LOGGER.info(f"{rank=} {i_step=} backward() finished")

            with timers["waiting"]:
                dist.barrier()

            with timers["update"]:
                optimizer.step()
                _LOGGER.info(f"{rank=} {i_step=} optimizer.step() finished")

                lr_scheduler.step()
                _LOGGER.info(f"{rank=} {i_step=} lr_scheduler.step() finished")

            with timers["waiting"]:
                dist.barrier()

            state["global_step"] += 1
            state["epoch_step"] += 1
            state["running_loss"] += outputs.loss.item()
            progress_bar.update(1)

            if state["global_step"] % args.log_freq == 0:
                mem = torch.cuda.memory_stats(device)
                info = {
                    f"lr": lr_scheduler.get_last_lr()[0],
                    f"running_loss": state["running_loss"] / args.log_freq,
                    f"epoch": state["epoch"],
                    f"epoch_progress": state["epoch_step"] / len(dataloader),
                    f"num_batches_remaining": len(dataloader) - i_step,
                    f"curr_memory_in_gb": 1e-9 * mem["allocated_bytes.all.current"],
                    f"peak_memory_in_gb": 1e-9 * mem["allocated_bytes.all.peak"],
                    f"time/total": sum(t.avg_elapsed_ms() for t in timers.values()),
                    **{
                        f"time/{k}": timer.avg_elapsed_ms()
                        for k, timer in timers.items()
                    },
                }
                _LOGGER.info(info)
                if rank == 0:
                    wandb.log(info, step=state["global_step"])

                torch.cuda.reset_peak_memory_stats(device)
                state["running_loss"] = 0
                for t in timers.values():
                    t.reset()

            if state["global_step"] % args.ckpt_freq == 0:
                dist.barrier()
                # NOTE: we have to call this on ALL ranks
                sharded_model_state, sharded_optimizer_state = get_state_dict(
                    model, optimizer, options=ckpt_opts
                )
                save(
                    dict(model=sharded_model_state, optimizer=sharded_optimizer_state),
                    checkpoint_id=exp_dir / "checkpoint",
                )
                if rank == 0:
                    torch.save(lr_scheduler.state_dict(), exp_dir / "lr_scheduler.pt")
                    with open(exp_dir / "state.json", "w") as fp:
                        json.dump(state, fp)
                dist.barrier()

        state["epoch_step"] = 0


def _load_and_preprocess_data(args, tokenizer, config):
    data = datasets.load_dataset(
        args.dataset_name, trust_remote_code=True, cache_dir=args.dataset_cache_root
    )

    column_names = data["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = data.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    seq_length = args.seq_length or tokenizer.model_max_length
    if seq_length > config.max_position_embeddings:
        seq_length = min(1024, config.max_position_embeddings)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        if total_length > seq_length:
            total_length = (total_length // seq_length) * seq_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + seq_length] for i in range(0, total_length, seq_length)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {seq_length}",
    )

    return lm_datasets["train"]


@contextmanager
def rank0_first():
    rank = dist.get_rank()
    if rank == 0:
        yield
    dist.barrier()
    if rank > 0:
        yield
    dist.barrier()


class LocalTimer:
    def __init__(self, device: torch.device):
        if device.type == "cpu":
            self.synchronize = lambda: torch.cpu.synchronize(device=device)
        elif device.type == "cuda":
            self.synchronize = lambda: torch.cuda.synchronize(device=device)
        self.measurements = []
        self.start_time = None

    def __enter__(self):
        self.synchronize()
        self.start_time = time.time()
        return self

    def __exit__(self, type, value, traceback):
        if traceback is None:
            self.synchronize()
            end_time = time.time()
            self.measurements.append(end_time - self.start_time)
        self.start_time = None

    def avg_elapsed_ms(self):
        return 1000 * (sum(self.measurements) / len(self.measurements))

    def reset(self):
        self.measurements = []
        self.start_time = None


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", default=None, required=True)
    parser.add_argument("--dataset-name", default=None, required=True)
    parser.add_argument("--model-name", default=None, required=True)
    parser.add_argument("--save-dir", default="../outputs")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num-epochs", default=100, type=int)
    parser.add_argument("--lr", default=3e-5, type=float)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--log-freq", default=100, type=int)
    parser.add_argument("--ckpt-freq", default=500, type=int)
    parser.add_argument("--dataset-cache-root", default="../.cache")
    parser.add_argument("--cpu-offload", default="on", choices=["on", "off"])
    parser.add_argument(
        "--bwd-prefetch",
        default="off",
        choices=["BACKWARD_PRE", "BACKWARD_POST", "off"],
    )
    parser.add_argument(
        "--activations",
        default="checkpoint",
        choices=["offload", "checkpoint", "in-memory"],
    )
    parser.add_argument("--seq-length", default=None, type=int)
    return parser


if __name__ == "__main__":
    main()
