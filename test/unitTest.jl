using DataFrames
using Base.Test
include("../src/utils.jl")

@testset "one hot encoding" begin
    dataA = DataFrame(x=[1, 2], y=[3, 4])
    @test OneHotEncode(dataA) == DataFrame(x_1=[1, 0], x_2=[0, 1], y_3=[1, 0], y_4=[0, 1])

    dataB = DataFrame(x=['a', 'b'], y=['c', 'd'])
    @test OneHotEncode(dataB) == DataFrame(x_a=[1, 0], x_b=[0, 1], y_c=[1, 0], y_d=[0, 1])

    dataC = DataFrame(x=['a', 'a'])
    @test OneHotEncode(dataC) == DataFrame(x_a=[1, 1])
end
