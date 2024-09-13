# How To Use
This is a template for machine learning (ML), or more specifically deep learning (DL), projects in Julia. The structure is what I find most convenient, and it should contain everything one needs to get started. Feel free to rename, rearrange, add, and remove anything you want. Let me know if I should add something, or if an existing part of the template is incorrect.

There are currently two main high-level frameworks for DL in Julia: Flux.jl and Lux.jl. A quick look around the web or through either documentation and one will find lots of discussion on the differences, pros/cons, etc., between the two. Presently, this template caters to Lux for its SciML integration. The hope is that a lot of the supporting code can be applied to either framework, and eventually I want to have support for both frameworks at a high level.

## Environment Setup

## Running Experiments
Use the following command to run an experiment:
```console
julia run.jl ...
```
If `logger` is set to `True` in the YAML config file, then the results of this experiment will be saved to `logs/<path to YAML file within ./experiments>`.

## Tips and Tricks

## Structure
- `core`: Model architectures, data loading, utilities, and core operators
- `data`: Dataset folders
- `experiments`: Experiment configuration files
  - `template.yaml`: Detailed experiment template
- `logs`: Experiment logs
- `run.jl`: Model training and testing script

## Documentation
The following is documentation for packages central to this template with descriptions of how they are used.

- Deep learning frameworks
    - [Lux.jl](https://lux.csail.mit.edu/): Default deep learning framework
    - [Flux.jl](https://fluxml.ai/Flux.jl/stable/): Alternative deep learning framework
    - [SimpleChains.jl](https://pumasai.github.io/SimpleChains.jl/stable/)
- [Julia Automatic Differentitation (AD)](https://juliadiff.org/)
    - [Zygote](https://fluxml.ai/Zygote.jl/stable/): Current default
    - [ForwardDiff](https://juliadiff.org/ForwardDiff.jl/stable/): Forward mode
    - [Enzyme](https://enzymead.github.io/Enzyme.jl/stable/): Future default?
- [](): Optmization algorithms
- [MLUtils.jl](https://juliaml.github.io/MLUtils.jl/stable/): Data processing functionality
- [JuliaLogging](https://julialogging.github.io/): Logging
    - [TensorBoardLogger.jl](https://julialogging.github.io/TensorBoardLogger.jl/stable/)
- [JLD2](https://juliaio.github.io/JLD2.jl/stable/): Data serialization

The following are packages one may be interested in using within this framework:

- [Scientific Machine Learning (SciML)](https://sciml.ai/)
    -
- [MLDatasets.jl](https://juliaml.github.io/MLDatasets.jl/stable/)
- [Metrics.jl](https://docs.juliahub.com/General/Metrics/stable/)