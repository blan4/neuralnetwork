module NeuralNetwork
  export NetworkData, Batch, feed_forward, train!

  type NetworkData
    biases::Array{Array{Float64,2},1}
    weights::Array{Array{Float64,2},1}
    sizes::Array{Int64,1}
  end

  type Batch
    input::Array{Float64,2}
    output::Array{Float64,2}
  end

  NetworkData(sizes::Array{Int,1}) = NetworkData(
    [randn(y, 1) for y in sizes[2:end]],
    [randn(y, x) for (x, y) in zip(sizes[1:end-1], sizes[2:end])],
    sizes)

  function feed_forward(nd::NetworkData, a::Array{Float64, 2})
    res = a
    for (b, w) in zip(nd.biases, nd.weights)
      res = sigmoid(w * res + b)
    end
    res
  end

  function train!(
    nd::NetworkData,
    train_data::Array{Batch,1},
    epochs::Int,
    batch_size::Int,
    learning_rate::Float64,
    test_data::Array{Batch,1},
    comparator::Function
  )
    for i in 1:epochs
      shuffle!(train_data)
      for k in 1:batch_size:length(train_data)
        update!(nd, train_data[k:k+batch_size-1], learning_rate / batch_size)
      end

      println("Epoch $i completed $(test(nd, test_data, comparator))")
    end
  end

  function update!(nd::NetworkData, batches::Array{Batch,1}, learning_rate::Float64)
    nabla_b = [zeros(b) for b in nd.biases]
    nabla_w = [zeros(w) for w in nd.weights]

    for batch in batches
      delta_nabla_b, delta_nabla_w = backprop(nd, batch)
      nabla_b = [nb + dnb for (nb, dnb) in zip(nabla_b, delta_nabla_b)]
      nabla_w = [nw + dnw for (nw, dnw) in zip(nabla_w, delta_nabla_w)]
    end

    nd.weights = [w - learning_rate * nw for (w, nw) in zip(nd.weights, nabla_w)]
    nd.biases = [b - learning_rate * nb for (b, nb) in zip(nd.biases, nabla_b)]
  end

  function backprop(nd::NetworkData, batch::Batch)
    nabla_b = [zeros(b) for b in nd.biases]
    nabla_w = [zeros(w) for w in nd.weights]

    activation::Array{Float64,2} = batch.input
    activations = []
    push!(activations, activation)
    zs::Array{Array{Float64,2},1} = []

    for (b, w) in zip(nd.biases, nd.weights)
      z = w * activation + b
      activation = sigmoid(z)
      push!(zs, z)
      push!(activations, activation)
    end

    delta = (activations[end] - batch.output) .* sigmoid_derivation(zs[end])
    nabla_b[end] = delta
    nabla_w[end] = delta * activations[end-1]'

    for l in 1:length(nd.sizes)-2
      z = zs[end-l]
      sp = sigmoid_derivation(z)
      delta = nd.weights[end-l+1]' * delta .* sp
      nabla_b[end-l] = delta
      nabla_w[end-l] = delta * activations[end-l-1]'
    end

    (nabla_b, nabla_w)
  end

  function test(nd::NetworkData, test_data::Array{Batch,1}, comparator::Function)
    s = 0
    for t in test_data
      res = feed_forward(nd, t.input)
      if (comparator(res, t.output)) s += 1 end
    end
    s / length(test_data)
  end

  function sigmoid(z)
    1 ./ (1 .+ exp(-z))
  end

  function sigmoid_derivation(z)
    s = sigmoid(z)
    s .* (1 .- s)
  end
end
