using DataFrames

"""
    oneHotEncode(data)

Do one-hot-encoding to the input data.

# Argument
- `data::DataFrame`: DataFrame from DataFrames package which contains only one-hot-encoding target

# Examples
```julia-OneHotEncode
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

            oneHotEncoded[row, col] = one(Int)
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
