import shutil
from pathlib import Path

import pydra
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from mytorch_lightning.apps.language_modelling import LM_Config, LM_Mydule
from mytorch_lightning.config import TrainingConfig
from mytorch_lightning.entry import train


def custom_collate_fn(batch):
    """Custom collate function that handles None values properly."""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    return {
        "input_ids": input_ids,
        "labels": None,  # Will be created by make_labels
        "loss_mask": None,  # No special masking
    }


class Config(LM_Config):
    def __init__(self):
        super().__init__()
        self.model_name: str = "Qwen/Qwen3-0.6B-Base"
        self.seq_len: int = 256  # Reduced from 1024 to get more samples
        self.num_proc: int = 64  # Number of processes for dataset operations


class FineTuneQwen(LM_Mydule):
    config: Config

    def __init__(self, config: Config):
        super().__init__(config)
        self._train_dataset = None
        self._val_dataset = None

    def create_model(self):
        device = f"cuda:{self.trainer.local_rank}"
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name, device_map=device
        )
        return model

    def get_logits(self, batch):
        outputs = self.model(batch["input_ids"])
        return outputs.logits

    def configure_training_dl(self, args: dict) -> dict:
        args["collate_fn"] = custom_collate_fn
        return args

    def configure_validation_dl(self, args: dict) -> dict:
        args["collate_fn"] = custom_collate_fn
        return args

    def train_data(self):
        if self._train_dataset is None:
            self._train_dataset, self._val_dataset = make_data(self.config)
        return self._train_dataset

    def val_data(self):
        if self._val_dataset is None:
            self._train_dataset, self._val_dataset = make_data(self.config)
        return self._val_dataset


class HFDatasetWrapper(Dataset):
    """Lightweight wrapper to ensure proper format for language modeling."""

    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "labels": None,  # Will be created by make_labels in LM_Mydule
            "loss_mask": None,  # No special masking for this test
        }


def make_data(config: Config):
    # Load wikitext dataset
    wikitext = load_dataset("Salesforce/wikitext", "wikitext-103-v1")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Define tokenization function
    def tokenize_function(examples):
        # Tokenize the text without truncation to get actual lengths
        tokenized = tokenizer(
            examples["text"],
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )
        return tokenized

    # Tokenize datasets
    print(f"Original train size: {len(wikitext['train'])}")
    print(f"Original validation size: {len(wikitext['validation'])}")

    tokenized_train = wikitext["train"].map(
        tokenize_function,
        batched=True,
        num_proc=config.num_proc,
        remove_columns=["text"],
        desc="Tokenizing train dataset",
    )

    tokenized_val = wikitext["validation"].map(
        tokenize_function,
        batched=True,
        num_proc=config.num_proc,
        remove_columns=["text"],
        desc="Tokenizing validation dataset",
    )

    # Filter out sequences that are too short
    def filter_by_length(example):
        return len(example["input_ids"]) >= config.seq_len

    filtered_train = tokenized_train.filter(
        filter_by_length,
        desc="Filtering train sequences by length",
        num_proc=config.num_proc,
    )

    filtered_val = tokenized_val.filter(
        filter_by_length,
        desc="Filtering validation sequences by length",
        num_proc=config.num_proc,
    )

    print(f"After filtering - train size: {len(filtered_train)}")
    print(f"After filtering - validation size: {len(filtered_val)}")

    # Truncate sequences to exact length
    def truncate_to_seq_len(example):
        example["input_ids"] = example["input_ids"][: config.seq_len]
        return example

    final_train = filtered_train.map(
        truncate_to_seq_len,
        desc="Truncating train sequences",
        num_proc=config.num_proc,
    )

    final_val = filtered_val.map(
        truncate_to_seq_len,
        desc="Truncating validation sequences",
        num_proc=config.num_proc,
    )

    # Wrap datasets to ensure proper format
    train_dataset = HFDatasetWrapper(final_train)
    val_dataset = HFDatasetWrapper(final_val)

    return train_dataset, val_dataset


class TestConfig(pydra.Config):
    def __init__(self):
        # Create training config with defaults
        self.training = TrainingConfig()
        self.training.ngpu = 1
        self.training.max_epochs = 1
        self.training.max_steps = 20
        self.training.train_batch_size = 2
        self.training.val_batch_size = 4
        self.training.gradient_accumulation_steps = 4
        self.training.lr = 1e-5
        self.training.lr_sched_type = "linear"
        self.training.warmup_steps = 10
        self.training.validation_step_interval = 10
        self.training.validation_limit_batches = 4
        self.training.gradient_clip_val = 1.0
        self.training.pin_memory = True
        self.training.persistent_workers = True
        self.training.train_shuffle = True
        self.training.val_shuffle = False
        self.training.num_workers = 8
        self.training.base_save_dir = "/scr/jbj/testing"
        self.training.save_every_n_steps = 10
        self.training.async_checkpointing = False

        # Model config
        self.model = Config()

        # Test-specific config
        self.cleanup_save_dir: bool = True


def main(config: TestConfig):
    # Create the mydule instance
    mydule = FineTuneQwen(config.model)

    # Use the train function from entry.py
    train(config.training, mydule)

    print("Training completed successfully!")

    save_dir = Path(config.training.base_save_dir)

    # List saved checkpoints
    if save_dir.exists():
        checkpoints = list(save_dir.glob("ckpt_*"))
        print(f"\nSaved checkpoints ({len(checkpoints)}):")
        for ckpt in sorted(checkpoints):
            print(f"  - {ckpt.name}")

        # Check if latest symlink exists
        latest = save_dir / "latest"
        if latest.exists():
            print(f"  - latest -> {latest.resolve().name}")

    # Final cleanup if requested
    if config.cleanup_save_dir and save_dir.exists():
        print(f"\nCleaning up save directory after test: {save_dir}")
        shutil.rmtree(save_dir)
        print("Cleanup completed.")


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    pydra.run(main)
