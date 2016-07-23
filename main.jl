include("neural_network.jl")

importall NeuralNetwork
using JLD

function load_train(path)
  data = readcsv(path)
  train_data = Batch[]
  for i in 2:size(data)[1]
    l = data[i,1]
    label = [if p == l 1.0 else 0.0 end for p in 0:9]'
    input = [d / 255 for d in data[i,2:end]]'
    push!(train_data, Batch(input', label'))
  end
  train_data
end

function load_test(path)
  data = readcsv(path)
  test_data = Array{Float64,2}[]
  for i in 2:size(data)[1]
    input = [d / 255 for d in data[i,1:end]]'
    push!(test_data, input')
  end
  test_data
end

function train()
  nd = NetworkData([784, 100, 10])
  println("Load data")
  train_data = load_train("data/train.csv")
  println("Start training")
  train!(nd, train_data, 100, 10, 3.0, train_data, ((got, expected) -> indmax(got) == indmax(expected)))
  println("Finish training")
  println("Saving")
  jldopen("data/nd.jld", "w") do file
    addrequire(file, NeuralNetwork)
    write(file, "nd", nd)
  end
  println("Saved")
  nd
end

function test(nd::NetworkData)
  println("Load test")
  test_data = load_test("data/test.csv")
  println("loaded")
  labels = Int[]
  println("Testing")
  for i in 1:length(test_data)
    l = feed_forward(nd, test_data[i])
    push!(labels, indmax(l)-1)
  end
  println("Finish")
  indexes = [i for i in 1:length(labels)]
  writecsv("data/submission.csv", [indexes labels])
  [indexes labels]
end

nd = train()
#nd::NetworkData = load("data/nd.jld", "nd")
test(nd)
