package org.seniorsigan.ml.neuralnetwork;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.List;
import java.util.Random;

import static org.seniorsigan.ml.neuralnetwork.Math.*;

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
        final DoubleMatrix res = new DoubleMatrix(input);
        zip(nd.weights, nd.biases, (w, b) -> {
            final DoubleMatrix newRes = sigmoid(Math.weighted(w, res, b));
            res.copy(newRes);
        });
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
            double[] res = feedForward(p.input.toArray());
            if (comparator.compare(res, p.output.toArray())) success++;
        }
        return success / (double)testBatch.size();
    }

    void update(final List<TrainPair> batch, final double learningRate) {
        DoubleMatrix[] nablaB = zeroClone(nd.biases);
        DoubleMatrix[] nablaW = zeroClone(nd.weights);

        for (final TrainPair tp: batch) {
            final Pair<DoubleMatrix[], DoubleMatrix[]> delta = backProp(nd, tp);
            nablaB = zipProduce(nablaB, delta.first, DoubleMatrix.class, DoubleMatrix::add);
            nablaW = zipProduce(nablaW, delta.second, DoubleMatrix.class, DoubleMatrix::add);
        }

        final Double eta = learningRate / batch.size();

        nd.weights = zipProduce(nd.weights, nablaW, DoubleMatrix.class, (w, nw) -> applyDelta(w, eta, nw));
        nd.biases = zipProduce(nd.biases, nablaB, DoubleMatrix.class, (b, nb) -> applyDelta(b, eta, nb));
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
