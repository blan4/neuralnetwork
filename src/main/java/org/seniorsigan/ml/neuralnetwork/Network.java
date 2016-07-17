package org.seniorsigan.ml.neuralnetwork;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.List;
import java.util.Random;

public class Network implements INetwork {
    private static final Logger log = LoggerFactory.getLogger(Network.class);

    private final NetworkData nd;
    private final Random rnd;
    private final Comparator comparator;

    public Network(final NetworkData networkData, final Comparator comparator) {
        this.rnd = new Random();
        this.comparator = comparator;
        this.nd = networkData;
    }

    @Override
    public double[] feedForward(final double[] input) {
        DoubleMatrix res = new DoubleMatrix(input);
        for (int i = 0; i < nd.biases.length; ++i) {
            res = Math.sigmoid(Math.weighted(nd.weights[i], res, nd.biases[i]));
        }
        return res.toArray();
    }

    @Override
    public void train(
            final List<TrainPair> trainData,
            final int epochs,
            final int batchSize,
            final double learningRate,
            final List<TrainPair> testData
    ) {
        for (int i = 0; i < epochs; ++i) {
            Collections.shuffle(trainData, rnd);
            for (int j = 0; j < trainData.size(); j += batchSize) {
                update(trainData.subList(j, j + batchSize), learningRate);
            }
            if (testData != null) {
                log.info("Epoch {} complete {}", i + 1, test(testData));
            } else {
                log.info("Epoch {} complete", i + 1);
            }
        }
    }

    @Override
    public double test(final List<TrainPair> testBatch) {
        int success = 0;
        for (final TrainPair p: testBatch) {
            double[] res = feedForward(p.input.data);
            if (comparator.compare(res, p.output.data)) success++;
        }
        return success / (double)testBatch.size();
    }

    private void update(final List<TrainPair> batch, final double learningRate) {
        final DoubleMatrix[] nablaB = Math.zeroClone(nd.biases);
        final DoubleMatrix[] nablaW = Math.zeroClone(nd.weights);

        for (final TrainPair tp: batch) {
            final Pair<DoubleMatrix[], DoubleMatrix[]> delta = backProp(tp);
            for (int i = 0; i < nablaB.length; ++i) {
                nablaB[i] = nablaB[i].add(delta.first[i]);
                nablaW[i] = nablaW[i].add(delta.second[i]);
            }
        }

        for (int i = 0; i < nd.biases.length; ++i) {
            nd.biases[i] = Math.applyDelta(nd.biases[i], learningRate / batch.size(), nablaB[i]);
            nd.weights[i] = Math.applyDelta(nd.weights[i], learningRate / batch.size(), nablaW[i]);
        }
    }

    /**
     *
     * @param batch
     * @return (nablaB, nablaW)
     */
    Pair<DoubleMatrix[], DoubleMatrix[]> backProp(final TrainPair batch) {
        final DoubleMatrix[] nablaB = Math.zeroClone(nd.biases);
        final DoubleMatrix[] nablaW = Math.zeroClone(nd.weights);

        final DoubleMatrix[] activations = new DoubleMatrix[nd.biases.length + 1];
        final DoubleMatrix[] zs = new DoubleMatrix[nd.biases.length];

        DoubleMatrix activation = batch.input;
        activations[0] = batch.input;

        for (int i = 0; i < nd.biases.length; ++i) {
            final DoubleMatrix z = Math.weighted(nd.weights[i], activation, nd.biases[i]);
            zs[i] = z;
            activation = Math.sigmoid(z);
            activations[i+1] = activation;
        }

        DoubleMatrix delta = Math.delta(activations[activations.length - 1], batch.output, zs[zs.length - 1]);
        nablaB[nablaB.length - 1] = delta;
        nablaW[nablaW.length - 1] = delta.mmul(activations[activations.length - 2].transpose());

        for (int l = 2; l < nd.layersCount; ++l) {
            final DoubleMatrix z = zs[zs.length - l];
            final DoubleMatrix sp = Math.sigmoidDerivation(z);
            delta = Math.mulMul(nd.weights[nd.weights.length - l + 1].transpose(), delta, sp);
            nablaB[nablaB.length - l] = delta;
            nablaW[nablaW.length - l] = delta.mmul(activations[activations.length - l - 1].transpose());
        }

        return new Pair<>(nablaB, nablaW);
    }

    /**
     * This class is used in the test() function to check
     * whether network recognize input right or not.
     */
    public interface Comparator {
        boolean compare(double[] got, double[] expected);
    }

    public NetworkData getNetworkData() {
        return nd;
    }
}
