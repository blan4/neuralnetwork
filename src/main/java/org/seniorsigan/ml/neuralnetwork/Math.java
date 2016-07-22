package org.seniorsigan.ml.neuralnetwork;

import org.jblas.DoubleMatrix;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;

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

    /**
     * Element-wise sigmoid derivation
     * d∂/dx = ∂*(1-∂), where ∂ = 1 / (1 + exp(-x))
     * @param x
     * @return
     */
    public static DoubleMatrix sigmoidDerivation(final DoubleMatrix x) {
        final DoubleMatrix res = new DoubleMatrix(x.rows, x.columns);
        for (int i = 0; i < x.rows; i++) {
            for (int j = 0; j < x.columns; j++) {
                res.put(i, j, sigmoidDerivation(x.get(i, j)));
            }
        }
        return res;
    }

    /**
     * d∂/dx = ∂*(1-∂), where ∂ = 1 / (1 + exp(-x))
     * @param x
     * @return
     */
    public static double sigmoidDerivation(final double x) {
        double s = sigmoid(x);
        return s * (1 - s);
    }

    /**
     *
     * @param w weights
     * @param a activation value
     * @param b biases
     * @return w*a+b
     */
    public static DoubleMatrix weighted(final DoubleMatrix w, final DoubleMatrix a, final DoubleMatrix b) {
        return (w.mmul(a)).add(b);
    }

    /**
     * Create new matrix of zeroes with shape equal to src matrix
     * @param src
     * @return matrix of zeroes
     */
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

    public static Pair<DoubleMatrix[], DoubleMatrix[]> backProp(final NetworkData nd, final TrainPair batch) {
        final DoubleMatrix[] nablaB = Math.zeroClone(nd.biases);
        final DoubleMatrix[] nablaW = Math.zeroClone(nd.weights);

        final List<DoubleMatrix> activations = new ArrayList<>();
        final List<DoubleMatrix> zs = new ArrayList<>();
        activations.add(batch.input);

        zip(nd.weights, nd.biases, (w, b) -> {
            final DoubleMatrix z = Math.weighted(w, activations.get(activations.size() - 1), b);
            zs.add(z);
            activations.add(Math.sigmoid(z));
        });

        DoubleMatrix delta = Math.delta(activations.get(activations.size() - 1), batch.output, zs.get(zs.size() - 1));
        nablaB[nablaB.length - 1] = delta;
        nablaW[nablaW.length - 1] = delta.mmul(activations.get(activations.size() - 2).transpose());

        for (int l = 2; l < nd.layersCount; ++l) {
            final DoubleMatrix z = zs.get(zs.size() - l);
            final DoubleMatrix sp = Math.sigmoidDerivation(z);
            delta = Math.mulMul(nd.weights[nd.weights.length - l + 1].transpose(), delta, sp);
            nablaB[nablaB.length - l] = delta;
            nablaW[nablaW.length - l] = delta.mmul(activations.get(activations.size() - l - 1).transpose());
        }

        return new Pair<>(nablaB, nablaW);
    }

    public static  <T, U> void zip(T[] a, U[] b, BiConsumer<T, U> func) {
        if (a.length != b.length) throw new IllegalArgumentException("Input arrays must be equal sized");
        for (int i = 0; i < a.length; i++) {
            func.accept(a[i], b[i]);
        }
    }

    public static  <T, U, R> R[] zipProduce(T[] t, U[] u, Class<R> clazz,  BiFunction<T, U, R> func) {
        if (t.length != u.length) throw new IllegalArgumentException("Input arrays must be equal sized");
        R[] res = (R[]) Array.newInstance(clazz, t.length);
        for (int i = 0; i < t.length; i++) {
            res[i] = func.apply(t[i], u[i]);
        }
        return res;
    }
}
