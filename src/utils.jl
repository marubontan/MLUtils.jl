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
