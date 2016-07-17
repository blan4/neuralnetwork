package org.seniorsigan.ml.neuralnetwork;

import org.jblas.DoubleMatrix;

import java.io.Serializable;

/**
 * Network's data storage. Can be easy converted to JSON.
 */
public class NetworkData implements Serializable {
    public final DoubleMatrix[] biases;
    public final DoubleMatrix[] weights;
    public final int layersCount;

    public NetworkData(DoubleMatrix[] biases, DoubleMatrix[] weights, int layersCount) {
        this.biases = biases;
        this.weights = weights;
        this.layersCount = layersCount;
    }

    public static NetworkData build(int[] sizes) {
        int layersCount = sizes.length;
        DoubleMatrix[] biases = new DoubleMatrix[layersCount - 1];
        DoubleMatrix[] weights = new DoubleMatrix[layersCount - 1];
        for (int i = 1; i < layersCount; ++i) {
            biases[i-1] = DoubleMatrix.randn(sizes[i]);
            weights[i-1] = DoubleMatrix.randn(sizes[i], sizes[i-1]);
        }
        return new NetworkData(biases, weights, layersCount);
    }
}
