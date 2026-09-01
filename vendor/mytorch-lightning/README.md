# Mytorch Lightning

This is a pure PyTorch ML training framework, heavily inspired by PyTorch Lightning. The goal is to:
 
- avoid as much boilerplate as possible
- always make it clear what's going on (e.g. no guessing about what data types)
- let us use PyTorch's newer feature set (e.g. torch.compile of the model and optimizer, FSDP2, TP, etc.)

More generally, it's about time we've owned our training code.

## Usage

To use the framework, you need to:

1. Create a Mydule class that inherits from `Mydule`.
2. Create torch datasets you need.
3. Instantiate a `Trainer` object (with the appropriate config) and call `train`.