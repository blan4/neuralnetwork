package org.seniorsigan.ml.neuralnetwork;

import org.jblas.DoubleMatrix;
import org.junit.Assert;
import org.junit.Test;

import static org.seniorsigan.ml.neuralnetwork.Math.backProp;
import static org.seniorsigan.ml.neuralnetwork.Math.weighted;

public class MathTest {
    double delta = 0.01;

    @Test
    public void Should_Weight() {
        DoubleMatrix expected = new DoubleMatrix(3,3, 10,10,10,10,10,10,10,20,10);
        DoubleMatrix w = new DoubleMatrix(3,2, 1,3,2,2,4,1);
        DoubleMatrix a = new DoubleMatrix(2,3, 2,1,1,1,4,2);
        DoubleMatrix b = new DoubleMatrix(3,3, 6,0,5,7,3,7,2,0,0);
        DoubleMatrix z = weighted(w, a, b);
        Assert.assertArrayEquals(expected.toArray(), z.toArray(), delta);
    }

    @Test
    public void Should_Backprop() {
        NetworkData nd = NetworkData.build(new int[]{2,3,1});
        nd.biases = new DoubleMatrix[]{
                new DoubleMatrix(3,1, 1,0.5,-0.3),
                new DoubleMatrix(1,1, 0.1)
        };
        nd.weights = new DoubleMatrix[]{
                new DoubleMatrix(3,2, 0.1,-1,-0.5,0.4,1,0.7),
                new DoubleMatrix(1,3, 0.5,0.3,-0.7)
        };
        Pair<DoubleMatrix[], DoubleMatrix[]> nabla = backProp(nd, TrainPair.build(new double[]{0.3,0.8}, new double[]{0.2}));
        DoubleMatrix[] nablaB = new DoubleMatrix[]{
                new DoubleMatrix(3,1, 0.007654, 0.005523, -0.016336),
                new DoubleMatrix(1,1, 0.093632)
        };
        DoubleMatrix[] nablaW = new DoubleMatrix[]{
                new DoubleMatrix(3,2, 0.00229616, 0.00165683, -0.00490085, 0.006123, 0.00441821, -0.01306894),
                new DoubleMatrix(1,3, 0.07435613, 0.06845064, 0.049388)
        };
        for(int i = 0; i < nablaB.length; ++i) {
            Assert.assertArrayEquals(nablaB[i].data, nabla.first[i].data, 0.00001);
        }
        for(int i = 0; i < nablaW.length; ++i) {
            Assert.assertArrayEquals(nablaW[i].data, nabla.second[i].data, 0.00001);
        }
    }
}
