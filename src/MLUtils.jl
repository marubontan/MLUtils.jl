module MLUtils

using Random, DataFrames

include("utils.jl")
export oneHotEncode, trainTestSplit, auc, brier

include("distance.jl")
export euclidean, minkowski
end
