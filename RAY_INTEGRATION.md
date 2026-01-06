# Ray Train Integration for Megatron-Bridge

This guide explains how to run Megatron-Bridge training with Ray Train for distributed orchestration.

## Prerequisites

### Required Packages

```bash
# Core dependencies
pip install ray[train]

# NVIDIA packages (required by Megatron-Bridge)
pip install --no-build-isolation 'transformer-engine[pytorch]'
pip install nvidia-modelopt
pip install nvidia-resiliency-ext
pip install megatron-energon
```

### Initialize Git Submodule

Megatron-Bridge uses a bundled version of Megatron-LM:

```bash
cd /path/to/Megatron-Bridge
git submodule update --init 3rdparty/Megatron-LM
```

## Quick Start

### Basic Usage (8 GPUs)

```bash
python scripts/training/finetune_decoder_ray.py \
    --hf_model_path meta-llama/Meta-Llama-3.1-8B \
    --num_workers 8 \
    --storage_path /mnt/local_storage
```

### Configuration Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--hf_model_path` | `meta-llama/Meta-Llama-3.1-8B` | HuggingFace model path |
| `--num_workers` | `8` | Total number of GPUs |
| `--tensor_parallel_size` | `2` | Tensor parallelism degree |
| `--pipeline_parallel_size` | `2` | Pipeline parallelism degree |
| `--train_iters` | `1000` | Training iterations |
| `--global_batch_size` | `64` | Global batch size |
| `--micro_batch_size` | `1` | Micro batch size per GPU |
| `--seq_length` | `2048` | Sequence length |
| `--learning_rate` | `5e-6` | Learning rate |
| `--storage_path` | `/mnt/cluster_storage` | Ray Train checkpoint storage |

### Parallelism Configuration

The total GPUs must satisfy: `num_workers = TP × PP × DP`

Example configurations for 8 GPUs:

| TP | PP | DP | Description |
|----|----|----|-------------|
| 2 | 2 | 2 | Balanced (default) |
| 2 | 4 | 1 | More pipeline stages |
| 4 | 2 | 1 | More tensor parallel |
| 1 | 4 | 2 | No tensor parallel |

```bash
# Custom parallelism
python scripts/training/finetune_decoder_ray.py \
    --num_workers 8 \
    --tensor_parallel_size 4 \
    --pipeline_parallel_size 2
```

## How It Works

### Key Integration Points

1. **Distributed Initialization**: Ray Train automatically initializes `torch.distributed` before calling the training loop. Megatron-Bridge detects this and skips its own initialization:

   ```
   torch distributed is already initialized, skipping initialization ...
   ```

2. **GPU Assignment**: Ray Train sets `CUDA_VISIBLE_DEVICES` for each worker. The script uses `external_gpu_device_mapping=True` to let Megatron-Bridge use device 0.

3. **Weight Loading**: Uses `AutoBridge.from_hf_pretrained()` with `load_weights=True` to load HuggingFace weights directly, bypassing the need for pre-converted Megatron checkpoints.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Ray Train (Coordinator)                  │
│                                                             │
│   ┌───────────────────────────────────────────────────┐    │
│   │              TorchTrainer                          │    │
│   │   - ScalingConfig(num_workers=8, use_gpu=True)    │    │
│   │   - train_loop_per_worker=train_loop              │    │
│   └───────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ Spawns 8 workers
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Worker Processes                          │
│                                                             │
│   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐         │
│   │ GPU 0   │ │ GPU 1   │ │ GPU 2   │ │ GPU 3   │         │
│   │ rank=0  │ │ rank=1  │ │ rank=2  │ │ rank=3  │         │
│   └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘         │
│        │           │           │           │               │
│   ┌────┴───────────┴───────────┴───────────┴────┐         │
│   │         TP=2 Group 0 | TP=2 Group 1          │         │
│   │              PP Stage 0                       │         │
│   └─────────────────────────────────────────────┘         │
│                                                             │
│   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐         │
│   │ GPU 4   │ │ GPU 5   │ │ GPU 6   │ │ GPU 7   │         │
│   │ rank=4  │ │ rank=5  │ │ rank=6  │ │ rank=7  │         │
│   └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘         │
│        │           │           │           │               │
│   ┌────┴───────────┴───────────┴───────────┴────┐         │
│   │         TP=2 Group 0 | TP=2 Group 1          │         │
│   │              PP Stage 1                       │         │
│   └─────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

### Code Flow

```
main()
  └─ TorchTrainer.fit()
       └─ train_loop(config) [on each worker]
            ├─ Add Megatron paths to sys.path
            ├─ create_megatron_config()
            │    ├─ AutoBridge.from_hf_pretrained(load_weights=True)
            │    ├─ Set TP, PP parallelism
            │    └─ Configure optimizer, dataset, etc.
            └─ pretrain(config, forward_step_func)
                 ├─ setup() - initializes model, optimizer
                 │    └─ Detects torch.distributed is initialized
                 └─ train() - runs training loop
```

## Model Support

The script supports any HuggingFace model that Megatron-Bridge supports:

- **Llama models**: `meta-llama/Meta-Llama-3.1-8B`, `meta-llama/Llama-3.2-1B`, etc.
- **Qwen models**: `Qwen/Qwen2-7B`, etc.
- **Other models**: Check Megatron-Bridge recipes for supported models

## Dataset

By default, the script uses the SQuAD dataset for finetuning:
- Automatically downloaded by HuggingFace datasets
- Configured via `default_squad_config()`
- Input/output format for question answering

## Troubleshooting

### Missing Dependencies

If you see `ModuleNotFoundError`, install the missing package:

```bash
# Common missing packages
pip install transformer-engine[pytorch]
pip install nvidia-modelopt
pip install nvidia-resiliency-ext
pip install megatron-energon
```

### Version Mismatch

If you see `No module named 'megatron.core.optimizer.layer_wise_optimizer'`:

```bash
# Initialize the bundled Megatron-LM submodule
git submodule update --init 3rdparty/Megatron-LM
```

### HuggingFace Authentication

For gated models like Llama:

```bash
huggingface-cli login
```

## Advanced Usage

### Custom Dataset

To use a custom dataset, modify `create_megatron_config()` to use a different dataset configuration instead of `default_squad_config()`.

### Resume from Checkpoint

The script saves checkpoints to `{storage_path}/megatron_outputs/checkpoints/`. To resume:

```bash
python scripts/training/finetune_decoder_ray.py \
    --hf_model_path meta-llama/Meta-Llama-3.1-8B \
    --num_workers 8 \
    --output_dir /previous/checkpoint/path
```

## Verified Test Run

The following test was run successfully on 8x H100 GPUs with Llama 3.1 8B:

```bash
python scripts/training/finetune_decoder_ray.py \
    --hf_model_path meta-llama/Meta-Llama-3.1-8B \
    --num_workers 8 \
    --train_iters 5 \
    --eval_interval 100 \
    --save_interval 100 \
    --storage_path /mnt/local_storage
```

### Key Log Output

```
# Workers started
Started training worker group of size 8:
- (ip=10.0.131.130, pid=1784723) world_rank=0, local_rank=0, node_rank=0
- (ip=10.0.131.130, pid=1784728) world_rank=1, local_rank=1, node_rank=0
...

# Ray Train's torch.distributed detected by Megatron-Bridge
torch distributed is already initialized, skipping initialization ...
> initialized tensor model parallel with size 2
> initialized pipeline model parallel with size 2
> setting random seeds to 5678 ...

# HuggingFace weights loaded
Loading from meta-llama/Meta-Llama-3.1-8B ━━━━━ 100% (195/195) LlamaBridge

# Training completed
[after training is done] datetime: 2026-01-06 14:21:39
saving checkpoint at iteration 5 to /mnt/local_storage/megatron_outputs/checkpoints
successfully saved checkpoint from iteration 5 to /mnt/local_storage/megatron_outputs/checkpoints

# Evaluation
Evaluating on 2048 samples
validation loss at iteration 5 | lm loss value: 1.039218E+00 | lm loss PPL: 2.827005E+00

Training completed successfully
Training finished. Result: Result(metrics=None, checkpoint=None, error=None,
    path='/mnt/local_storage/megatron_ray_1df88305', ...)
```

## References

- [Ray Train Documentation](https://docs.ray.io/en/latest/train/train.html)
- [Megatron-Bridge Documentation](https://github.com/NVIDIA/Megatron-Bridge)
- [Megatron-LM Documentation](https://github.com/NVIDIA/Megatron-LM)
