function OneHotEncode(data::DataFrame)

    columns = names(data)
    rowSize = nrow(data)

    output = DataFrame()
    for (i,column) in enumerate(columns)

        uniqueValues = sort(unique(data[column]))

        # make label
        columnStr = string(column)
        label = Symbol.([columnStr * "_" * string(val) for val in uniqueValues])

        oneHotEncoded = zeros(rowSize, length(uniqueValues))
        for (row,val) in enumerate(data[column])
            col = find(uniqueValues .== val)

            oneHotEncoded[row, col] = 1
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
