#=
Data processing functionality for Lux models in Julia.

References:
    - https://juliaml.github.io/MLUtils.jl/stable/api/#MLUtils.DataLoader
    - https://lux.csail.mit.edu/dev/tutorials/intermediate/3_HyperNet
    
Author: Cooper Simpson
=#

using Random, Threads
using JLD2
using MLUtils: DataLoader, splitobs, numobs, getobs!, getobs

#=
Custom datset type
=#
mutable struct Dataset{}

end

function Base.length(data::Dataset)
end

function Base.getindex(data::Dataset, idx::I) where {I}
end

function numobs(data::Dataset)
end

function getobs!(buffer::T, data::Dataset, idx::I) where{T, I}
end

function getobs(data::Dataset, idx::I) where{I}
end

#=
Outer constructor for building Dataset object
=#
function Dataset()

end

#=
Builds a DataLoader
=#
function get_dataloader()

    #build dataset
    dset = Dataset()

    #possibly partition dataset
    splitobs(dset)

    #check num threads if parallel loading enabled
    if parallel & Threads.nthreads() == 1
        @warn "Parallel dataloading requested, but Julia was started with only one thread."
    end

    return DataLoader(dset)

end