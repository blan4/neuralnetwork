package org.seniorsigan.ml.neuralnetwork;

import org.jblas.DoubleMatrix;

public class TrainPair {
    public final DoubleMatrix input;
    public final DoubleMatrix output;

    private TrainPair(DoubleMatrix input, DoubleMatrix output) {
        if (!input.isColumnVector() || !output.isColumnVector()) {
            throw new IllegalArgumentException("Input and output matrix must be column vector");
        }
        this.input = input;
        this.output = output;
    }

    public static TrainPair build(double[] input, double[] output) {
        return new TrainPair(new DoubleMatrix(input), new DoubleMatrix(output));
    }
}
