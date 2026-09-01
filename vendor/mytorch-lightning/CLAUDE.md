# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

mytorch-lightning is a pure PyTorch ML training framework inspired by PyTorch Lightning. It minimizes boilerplate while maintaining clarity and leveraging modern PyTorch features (torch.compile, FSDP2, etc.). Supports both single-node and multi-node distributed training.

## Common Development Commands

### Installation
```bash
# Development installation
pip install -e ".[dev]"

# Basic installation
pip install -e .
```

### Testing
```bash
# Run all tests
pytest mytorch_lightning/test/

# Run specific test
pytest mytorch_lightning/test/test_lm.py

# Run the language model training test
python mytorch_lightning/test/test_lm.py
```

### Code Quality
```bash
# Format code
ruff format .

# Lint code
ruff check .

# Fix linting issues
ruff check --fix .
```

## Architecture Overview

### Core Components

1. **Mydule** (`mydule.py`, 71 lines): Base module class extending LoggingModule
   - Key methods: `create_model()`, `initialize_model()`, `training_step()`, `validation_step()`
   - Subclass this for custom models
   - Returns loss from training/validation steps
   - Provides `configure_optimizer()` hook (defaults to AdamW)
   - Configure DataLoader args via `configure_training_dl()` and `configure_validation_dl()`

2. **Trainer** (`trainer.py`, 813 lines): Main training orchestrator
   - Handles distributed training (DDP, FSDP)
   - Manages training loops, validation, checkpointing
   - Supports async checkpointing with process groups
   - Takes datasets directly (not DataLoaders) - creates them internally
   - Automatic mixed precision (AMP) support
   - Gradient accumulation and clipping
   - Learning rate scheduling via callbacks

3. **TrainingConfig** (`config.py`, 87 lines): Central configuration dataclass using pydra.Config
   - All training parameters in one place
   - Multi-node support: `nodes`, `node_rank`, `master_addr`, `master_port`
   - `ngpu` now refers to GPUs per node (not total GPUs)
   - Supports wandb integration via WandbConfig
   - Configurable checkpointing (sync/async, pruning old checkpoints)
   - Optimizer configuration (AdamW with foreach/fused options)
   - Data loading parameters (workers, pinning, shuffling)

4. **Logging System** (`logging.py`, 205 lines): Distributed-aware logging
   - `LoggingModule`: Base class with `log()` method for metrics
   - `Reducer`: Handles metric reduction across devices (mean, sum, max, min)
   - Supports on_step/on_epoch logging with automatic aggregation
   - Progress bar integration

5. **Callback System** (`callback.py`, 49 lines): Extensible training hooks
   - Implement custom callbacks by subclassing `Callback`
   - Hooks: `on_train_start/end`, `on_train_epoch_start/end`, `on_train_step_start/end`
   - Validation hooks: `on_validation_epoch_start/end`, `on_validation_step_start/end`
   - Optimizer hooks: `on_before/after_optimizer_step`

6. **Language Modeling** (`apps/language_modelling.py`, 194 lines): LM-specific utilities
   - `LM_Mydule`: Base class for language models with token tracking
   - `make_labels()`: Creates labels from input_ids with proper masking
   - Automatic metrics: loss, accuracy, perplexity, tokens/sec
   - Z-loss regularization support
   - `initialize_linears_and_embeddings()`: Weight initialization helper

7. **Learning Rate Schedulers** (`lr_schedule.py`, 88 lines): Built-in LR scheduling
   - `ConstantLR_Scheduler`: Constant LR with warmup
   - `LinearLR_Scheduler`: Linear decay from peak to final LR
   - `CosineLR_Scheduler`: Cosine annealing schedule
   - All support warmup period

8. **Entry Point** (`entry.py`, 49 lines): Multi-GPU launch utilities
   - `train()`: Main entry point that handles single/multi-GPU
   - `launch_procs_and_train()`: Spawns processes for distributed training
   - Automatic port finding for distributed setup

### Usage Example

```python
import pydra
from mytorch_lightning.apps.language_modelling import LM_Config, LM_Mydule
from mytorch_lightning.entry import train
from mytorch_lightning.config import TrainingConfig

class Config(LM_Config):
    def __init__(self):
        super().__init__()
        self.model_name: str = "gpt2"
        self.seq_len: int = 1024

class MyLM(LM_Mydule):
    def __init__(self, config: Config):
        super().__init__(config)
    
    def create_model(self):
        # Return your model (can use device_map for efficiency)
        device = f"cuda:{self.trainer.global_rank}"
        return AutoModelForCausalLM.from_pretrained(
            self.config.model_name, device_map=device
        )
    
    def get_logits(self, batch):
        # Return logits from model
        outputs = self.model(batch["input_ids"])
        return outputs.logits
    
    def train_data(self):
        # Return PyTorch Dataset
        return train_dataset
    
    def val_data(self):
        # Return PyTorch Dataset or None
        return val_dataset

class ScriptConfig(pydra.Config):
    def __init__(self):
        self.training = TrainingConfig()
        self.training.max_epochs = 3
        self.training.train_batch_size = 8
        self.model = Config()

def main(config: ScriptConfig):
    mydule = MyLM(config.model)
    train(config.training, mydule)

if __name__ == "__main__":
    pydra.run(main)
```

### Data Format

For language modeling, datasets should return dictionaries with:
- `input_ids`: Torch tensor of token IDs (required)
- `labels`: None (auto-generated) or custom labels (optional)
- `loss_mask`: None or boolean mask for loss computation (optional)

The framework will automatically:
- Create labels by shifting input_ids (for next-token prediction)
- Apply loss_mask if provided to exclude certain tokens from loss
- Handle padding tokens (masked with value -100)

### Key Design Patterns

1. **Configuration Management**: Uses `pydra-config` library
   ```python
   import pydra
   
   class ScriptConfig(pydra.Config):
       def __init__(self):
           self.batch_size: int = 32
           self.learning_rate: float = 1e-4
   
   def main(config: ScriptConfig):
       # Use config.batch_size, config.learning_rate
       pass
   
   if __name__ == "__main__":
       pydra.run(main)
   ```
   Run with: `python script.py batch_size=64 learning_rate=3e-4`

2. **Dataset Integration**: 
   - Framework expects PyTorch Dataset objects, not DataLoaders
   - Can wrap HuggingFace datasets with a lightweight wrapper
   - Trainer creates DataLoaders internally with config parameters
   - Distributed sampling handled automatically via DistributedSampler

3. **Distributed Training**: Handled automatically by entry.py
   - Single-GPU: Direct execution
   - Multi-GPU: Spawns processes with proper env vars
   - DDP wrapper applied automatically based on strategy
   - Logging synchronized across ranks

4. **Checkpointing**:
   - Saves model and optimizer state using torch.distributed.checkpoint
   - Async checkpointing option for large models
   - Automatic pruning of old checkpoints
   - Latest checkpoint symlinked for easy access

5. **Metric Logging**:
   - Use `self.log()` in Mydule methods
   - Automatic reduction across devices
   - Progress bar integration
   - Wandb support built-in

### Important Notes

- The framework prioritizes clarity over magic - data types are explicit
- Designed for modern PyTorch features (compile, FSDP2, etc.)
- Callbacks provide extensibility without modifying core code
- Logging is distributed-aware with automatic synchronization
- Supports both epoch-based and step-based training
- Always pass datasets to Trainer, not DataLoaders
- Trainer handles device placement automatically (don't move data in Mydule)
- Use absolute paths for `base_save_dir` in TrainingConfig
- Set multiprocessing start method to "spawn" for multi-GPU training

### Multi-Node Training

Launch training on multiple nodes without torchrun:

```bash
# Node 0 (master):
python train.py training.nodes=2 training.node_rank=0 training.master_addr=192.168.1.100 training.master_port=10210

# Node 1:
python train.py training.nodes=2 training.node_rank=1 training.master_addr=192.168.1.100 training.master_port=10210
```

Key multi-node parameters:
- `nodes`: Total number of nodes
- `node_rank`: Rank of the current node (0-indexed)
- `master_addr`: IP address of the master node (node 0)
- `master_port`: Port for inter-node communication (default: 10210)
- `ngpu`: Number of GPUs per node (not total GPUs)

### Common Patterns

1. **Custom Collate Functions**: Define in Mydule's configure_*_dl methods
2. **Mixed Precision**: Set `amp_dtype="bfloat16"` or `"float16"` in TrainingConfig
3. **Gradient Accumulation**: Set `gradient_accumulation_steps` > 1
4. **Learning Rate Schedule**: Automatically added based on `lr_sched_type`
5. **Validation**: Set `validation_step_interval` or `validation_epoch_interval`
6. **Early Stopping**: Use `max_steps` to limit training