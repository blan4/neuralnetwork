package org.seniorsigan.ml.neuralnetwork;

import org.jblas.DoubleMatrix;
import org.junit.Assert;
import org.junit.Test;

public class MathTest {
    double delta = 0.01;

    @Test
    public void Should_Weight() {
        DoubleMatrix expected = new DoubleMatrix(3,3, 10,10,10,10,10,10,10,20,10);
        DoubleMatrix w = new DoubleMatrix(3,2, 1,3,2,2,4,1);
        DoubleMatrix a = new DoubleMatrix(2,3, 2,1,1,1,4,2);
        DoubleMatrix b = new DoubleMatrix(3,3, 6,0,5,7,3,7,2,0,0);
        DoubleMatrix z = Math.weighted(w, a, b);
        Assert.assertArrayEquals(expected.toArray(), z.toArray(), delta);
    }
}
