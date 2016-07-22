include("neural_network.jl")

importall NeuralNetwork

function read_data(path)
  data = readcsv("train.csv")
  train_data = Batch[]
  for i in 2:size(data)[1]
    l = data[i,1]
    label = [if p == l 1.0 else 0.0 end for p in 0:9]'
    input = [d / 255 for d in data[i,2:end]]'
    push!(train_data, Batch(input', label'))
  end
  train_data
end

function mnist_test()
  println("Load data")
  train_data = read_data("train.csv")
  nd = NetworkData([784, 30, 10])
  println("Start training")
  train!(nd, train_data[1:30000], 1000, 10, 3.0, train_data[30001:end], ((got, expected) -> indmax(got) == indmax(expected)))
  println("Finish training")
end

mnist_test()



########
# TEST #
########

function xorTest()
  nd = NetworkData([2,4,1])
  data = [Batch([0.0 0.0]', [0.0]'), Batch([1.0 1.0]', [0.0]'), Batch([1.0 0.0]', [1.0]'), Batch([0.0 1.0]', [1.0]')]
  println("Start training")
  train!(nd, data, 1000, 2, 0.1, data, ((got, expected) -> round(got) == expected))
  println("Finish training")
end

# xorTest()
