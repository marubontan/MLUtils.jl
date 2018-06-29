using DataFrames


"""
    oneHotEncode(data)

Do one-hot-encoding to the input data.

# Argument
- `data::DataFrame`: DataFrame from DataFrames package which contains only one-hot-encoding target

# Examples
```julia-oneHotEncode
julia> dataA = DataFrame(x=[1, 2], y=[3, 4])
2×2 DataFrames.DataFrame
│ Row │ x │ y │
├─────┼───┼───┤
│ 1   │ 1 │ 3 │
│ 2   │ 2 │ 4 │

julia> oneHotEncode(dataA)
2×4 DataFrames.DataFrame
│ Row │ x_1 │ x_2 │ y_3 │ y_4 │
├─────┼─────┼─────┼─────┼─────┤
│ 1   │ 1   │ 0   │ 1   │ 0   │
│ 2   │ 0   │ 1   │ 0   │ 1   │
```
"""
function oneHotEncode(data::DataFrame)

    columns = names(data)
    rowSize = nrow(data)

    output = DataFrame()
    for (i,column) in enumerate(columns)

        sortedUniqueValues = sort(unique(data[column]))

        # make label
        columnStr = string(column)
        label = Symbol.([columnStr * "_" * string(val) for val in sortedUniqueValues])

        oneHotEncoded = zeros(Int, rowSize, length(sortedUniqueValues))
        for (row,val) in enumerate(data[column])

            col = find(sortedUniqueValues .== val)

            oneHotEncoded[row, col] .= one(Int)
        end
        oneHotEncodedDF = DataFrame(oneHotEncoded)

        # attach column names
        names!(oneHotEncodedDF, label)

        if i == 1
            output = oneHotEncodedDF
        else
            output = hcat(output, oneHotEncodedDF)
        end
    end

    return output
end


"""
    trainTestSplit(data; trainSize=0.5, shuffle=true)

Split data into train and test ones with specified rate.

# Argument
- `data::DataFrame`: target data to split
- `trainSize::Float64`: assigned rate for train data(0 < trainSize < 1)
- `shuffle::Bool`: if true, the data is shuffled before splited

# Examples
```julia-trainTestSplit
julia> dataA = DataFrame(x=[1, 2], y=[3, 4])
2×2 DataFrames.DataFrame
│ Row │ x │ y │
├─────┼───┼───┤
│ 1   │ 1 │ 3 │
│ 2   │ 2 │ 4 │

julia> trainTestSplit(dataA)
(1×2 DataFrames.DataFrame
│ Row │ x │ y │
├─────┼───┼───┤
│ 1   │ 2 │ 4 │, 1×2 DataFrames.DataFrame
│ Row │ x │ y │
├─────┼───┼───┤
│ 1   │ 1 │ 3 │)
```
"""
function trainTestSplit(data::DataFrame; trainSize=0.5, shuffle=true)

    @assert 0 < trainSize < 1

    rowNum = nrow(data)
    rowIndex = collect(1:rowNum)

    shuffle && shuffle!(rowIndex)
    threshold = Int(ceil(rowNum * trainSize))

    return (data[rowIndex[1:threshold], :], data[rowIndex[threshold + 1:rowNum], :])
end


"""
	auc(yTruth, yScore)

Calculate AUC score to ROC curve.

# Argument
- `yTruth::Array`: The Array composed of (0, 1) which indicate the class
- `yScore::Array`: The predicted scores

# Examples
```julia-auc
julia> yTruth = [1, 0, 1]
3-element Array{Int64,1}:
 1
 0
 1

julia> yScore = [0.2, 0.3, 0.8]
3-element Array{Float64,1}:
 0.2
 0.3
 0.8

julia> auc(yTruth, yScore)
0.5
```
"""
function auc(yTruth::Array{Int}, yScore::Array{Float64})

	dIndex = find(1 .== yTruth)
	dnIndex = find(0 .== yTruth)
	score = 0.0

	for ydn in dnIndex

		for yd in dIndex

			if yScore[yd] > yScore[ydn]
				score += 1
			elseif yScore[ydn] == yScore[yd]
				score += 0.5
			end

		end
	end

	return score / (length(dIndex) * length(dnIndex))
end


"""
    brier(yTruth, yScore)

Calculate brier score.

# Argument
- `yTruth::Array`: The Array composed of (0, 1) which indicate the class
- `yScore::Array`: The predicted scores

# Examples
```julia-brier
julia> yTruth = [1, 0, 1]
3-element Array{Int64,1}:
 1
 0
 1

julia> yScore = [0.2, 0.3, 0.8]
3-element Array{Float64,1}:
 0.2
 0.3
 0.8

julia> brier(yTruth, yScore)
0.25666666666666665
```
"""
function brier(yTruth::Array{Int}, yScore::Array{Float64})

    n = length(yTruth)

    errorSum = zero(1)
    for iter in zip(yTruth, yScore)
        errorSum += abs2(iter[2] - iter[1])
    end

    return errorSum / n
end
