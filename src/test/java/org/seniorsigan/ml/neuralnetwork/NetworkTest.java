package org.seniorsigan.ml.neuralnetwork;

import org.jblas.DoubleMatrix;
import org.junit.Assert;
import org.junit.Test;
import org.junit.experimental.theories.suppliers.TestedOn;

import java.lang.Math;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

public class NetworkTest {
    double delta = 0.01;

    @Test
    public void Should_TrainXOR() {
        final Network nn = new Network(NetworkData.build(new int[]{2,10,1}), (got, expected) -> {
            System.out.println("Got :" + got[0] + " Expected :" + expected[0]);
            for (int i = 0; i < got.length; i++) {
                if(!Objects.equals(
                        Math.round(got[i]),
                        Math.round(expected[i]))
                ) {
                    return false;
                }
            }
            return true;
        });
        final TrainPair[] trainDataSet = new TrainPair[]{
                TrainPair.build(new double[]{0,0}, new double[]{0}),
                TrainPair.build(new double[]{0,1}, new double[]{1}),
                TrainPair.build(new double[]{1,0}, new double[]{1}),
                TrainPair.build(new double[]{1,1}, new double[]{0})
        };
        nn.train(Arrays.asList(trainDataSet), 150, 4, 2, Arrays.asList(trainDataSet));
        Assert.assertEquals(1.0, nn.test(Arrays.asList(trainDataSet)), 0.1);
    }

    @Test
    public void Should_Update() {
        NetworkData nd = NetworkData.build(new int[]{2,3,1});
        nd.biases = new DoubleMatrix[]{
                new DoubleMatrix(3,1, 1,0.5,-0.3),
                new DoubleMatrix(1,1, 0.1)
        };
        nd.weights = new DoubleMatrix[]{
                new DoubleMatrix(3,2, 0.1,-1,-0.5,0.4,1,0.7),
                new DoubleMatrix(1,3, 0.5,0.3,-0.7)
        };
        Network nn = new Network(nd, (got, expected) -> true);
        List<TrainPair> train = new ArrayList<>();
        train.add(TrainPair.build(new double[]{0.3,0.8}, new double[]{0.2}));
        nn.update(train, 3.0);
        System.out.println("Hello");
    }
}
