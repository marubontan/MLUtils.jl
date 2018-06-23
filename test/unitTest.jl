using DataFrames
using Base.Test
include("../src/utils.jl")

@testset "one hot encoding" begin
    dataA = DataFrame(x=[1, 2], y=[3, 4])
    @test oneHotEncode(dataA) == DataFrame(x_1=[1, 0], x_2=[0, 1], y_3=[1, 0], y_4=[0, 1])

    dataB = DataFrame(x=['a', 'b'], y=['c', 'd'])
    @test oneHotEncode(dataB) == DataFrame(x_a=[1, 0], x_b=[0, 1], y_c=[1, 0], y_d=[0, 1])

    dataC = DataFrame(x=['a', 'a'])
    @test oneHotEncode(dataC) == DataFrame(x_a=[1, 1])
end

@testset "auc" begin
    yA = [1, 0, 1]
    yScoreA = [0.2, 0.3, 0.8]
    @test auc(yA, yScoreA) == 0.5

    yB = [1, 1, 0]
    yScoreB = [0.8, 0.8, 0.1]
    @test auc(yB, yScoreB) == 1.0

    yC = [1, 0, 0]
    yScoreC = [0.3, 0.6, 0.5]
    @test auc(yC, yScoreC) == 0.0
end
