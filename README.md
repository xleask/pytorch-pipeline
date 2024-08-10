# Base PyTorch Pipeline

Base/example functionality for training and evaluating PyTorch models, supporting adversarial training and programmatic creation of model architectures for quick experimentation.

This pipeline is designed to be built upon for project-specific requirements. Please fork this repo for specific projects.

## Requirements

Python >3.10

View `requirements.txt` for external packages. Note, `pip install -r requirements.txt` may not work for installing `torch` and `torchvision`, it is recommended to install them separately to ensure compatability with the running workstation.

## Quick Start

There are four main functionalities supported by this pipeline:
1. Model **training** through `train.py`
2. Trained model **evaluation** through `evaluate.py`
3. Trained model **training evaluation** through `evaluate_training.py`
4. **JIT-tracing** a model for profiling through `jit_trace_model.py`

Each of the above are defined by YAML config files in a model-specific directory within `configs`.
Model architectures and their corresponding model flow are defined in their model-specific directory within `models`.

A toy example is provided that covers the basic functionality. Use the following commands to perform the above four functionalities.

**Train** basic GAN:
```
python train.py -t configs/example_model/example_training_cfg.yaml
```
or training using [DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) via `torchrun`:
```
torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py -t configs/example_model/example_multigpu_training_cfg.yaml
```
where the value for `--nproc_per_node` must match the length of the GPU list in the training config.

**Evaluate GAN training** (specify trained model checkpoint in YAML):
```
python evaluate_training.py -te configs/example_model/example_training_evaluation_cfg.yaml
```

**Evaluate trained GAN output** (specify trained model checkpoint in YAML):
```
python evaluate.py -t configs/example_model/example_training_cfg.yaml -e configs/example_model/example_evaluation_cfg.yaml
```

**JIT-trace** trained model (specify trained model checkpoint):
```
python jit_trace_model.py -t configs/example_model/example_training_cfg.yaml -c <recent_trained_model.pth> -i 1 1 4 4
```

## Model Design

Models are specified by their model architecture and their model flow that live in the same subdirectory under `models`.

### Model Architecture

The model architecture is defined in a YAML file (view an [example architecture](models/example_model/example_model_architecture.yaml) for reference) and supports built-in torch layers and arguments, as well as custom layers that must be defined in `layers.py`.

There are two main parts to designing an architecture:
1. Defining high-level `model_components`. Typically, these components are flow-independent, such as in the case of a GAN where the generator and discriminator act as standalone networks.
  - Each model component must have one or more `submodels`. Submodels must be defined in order of their dependence, i.e., an earlier submodel must not depend on the output of a later submodel. Submodels allow for flexibility in model flow.
2. Defining low-level `submodels`. Each `submodel` in `model_components` must be defined in the YAML with the following keys:
  - `submodel_inputs`: Either directly from the training batch (`input_data`) or from the output of a submodel (e.g., `{gen_cnn: 'output'}`).
  - `layer_name_prefix`: String to prepend all layer names. Used for clarity and skip connections.
  - `layers`: Dictionary of layer name (defined in torch or `layers.py`) and corresponding keyword arguments.
  - `skip_connections`: Supports skip connections within and across submodels. Dictionary of submodel layer name keys with values that are dictionaries to future layers in other submodels. E.g., `{gen_cnn_Conv2d_0: {'future_layers': ['gen_dec_ReLU_4'], connection_types: ['add']}}`

### Model Flow

Model flows are defined in a regular torch `nn.Module` class in a model-specific directory (e.g., [here](models/example_model/example_model.py)). Model components are read from the model architecture and accessed through `self.model_components`. For example, inputs can be mapped in the forward method via:
```
new_skip_dict = {}
generator = self.model_components['generator']
gen_outputs = generator['gen_cnn'](inputs, new_skip_dict)
# Reset global skip output buffer.
self.global_skip_outputs.clear()
```
Note, `new_skip_dict` should always be instantiated on a new pass through the model. `global_skip_outputs` handles skip connections across submodels.

## Datasets

A regular torch dataset function is defined in `dataset.py`, which should be adapted for collecting batches. Functionality is provided via `dataset_base.py` and `dataset_functions.py` for reading, postprocessing, and batch-transforming data. Custom functions should be defined in `dataset_functions.py` and specified to be used in the training configs.

## Loss

A regular torch loss function is defined in `loss.py`.

## Training

For training, the only input is a model-specific training config (see, e.g., [here](configs/example_model/example_training_cfg.yaml)). The config specifies:
- (Generator) training/optimizer details
- Discriminator training/optimizer details, if applicable
- Model architecture, flow, and checkpoint, if applicable
- Loss function and associated arguments (e.g., weightings)
- Saving parameters
- Dataset files, function, and dataset augmentation options

The goal is to avoid any modification to the training loop in `train.py` but, of course, this is inevitable for certain projects. The loop supports adversarial training.

## Evaluation

Similar to training, model output evaluation and model training evaluation configs can be used. This is project-specific. View the example as a guide.

## JIT-Tracing

For power/latency profiling, `jit_trace_model.py` provides a JIT-traced model and an example raw input to the model to be tested on device.
