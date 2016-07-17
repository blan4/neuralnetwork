package org.seniorsigan.ml.neuralnetwork;

import java.util.List;

public interface INetwork {
    double[] feedForward(double[] input);
    void train(List<TrainPair> trainData, int epochs, int batchSize, double learningRate, List<TrainPair> testData);
    double test(List<TrainPair> testBatch);
}
