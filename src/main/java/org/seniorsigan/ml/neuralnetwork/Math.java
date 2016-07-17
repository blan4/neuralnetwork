package org.seniorsigan.ml.neuralnetwork;

import org.jblas.DoubleMatrix;

public class Math {
    public static double sigmoid(final double z) {
        return 1.0 / (1.0 + java.lang.Math.exp(-z));
    }

    public static DoubleMatrix sigmoid(final DoubleMatrix z) {
        final DoubleMatrix res = new DoubleMatrix(z.rows, z.columns);
        for (int i = 0; i < z.rows; i++) {
            for (int j = 0; j < z.columns; j++) {
                res.put(i, j, sigmoid(z.get(i, j)));
            }
        }
        return res;
    }

    public static DoubleMatrix sigmoidDerivation(final DoubleMatrix s) {
        final DoubleMatrix res = new DoubleMatrix(s.rows, s.columns);
        for (int i = 0; i < s.rows; i++) {
            for (int j = 0; j < s.columns; j++) {
                res.put(i, j, sigmoidDerivation(s.get(i, j)));
            }
        }
        return res;
    }

    public static double sigmoidDerivation(final double z) {
        double s = sigmoid(z);
        return s * (1 - s);
    }

    // w*a+b
    public static DoubleMatrix weighted(final DoubleMatrix w, final DoubleMatrix a, final DoubleMatrix b) {
        return (w.mmul(a)).add(b);
    }

    public static DoubleMatrix[] zeroClone(final DoubleMatrix[] src) {
        final DoubleMatrix[] out = new DoubleMatrix[src.length];
        for (int i = 0; i < src.length; ++i) {
            out[i] = DoubleMatrix.zeros(src[i].rows, src[i].columns);
        }
        return out;
    }

    // a - b * v
    public static DoubleMatrix applyDelta(final DoubleMatrix a, final double v, final DoubleMatrix b) {
        return a.sub(b.mul(v));
    }

    // m1 * m2 x m3
    public static DoubleMatrix mulMul(final DoubleMatrix m1, final DoubleMatrix m2, final DoubleMatrix m3) {
        return (m1.mmul(m2)).mul(m3);
    }

    // (a-o)*z
    public static DoubleMatrix delta(final DoubleMatrix activation, final DoubleMatrix output, final DoubleMatrix z) {
        return (activation.sub(output)).mul(Math.sigmoidDerivation(z));
    }
}
