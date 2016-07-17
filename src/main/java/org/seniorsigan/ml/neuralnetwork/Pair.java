package org.seniorsigan.ml.neuralnetwork;

public class Pair<T1, T2> {
    public final T1 first;
    public final T2 second;

    public Pair(T1 first, T2 second) {
        if (first == null || second == null) {
            throw new IllegalArgumentException("Pair elements can't be null");
        }
        this.first = first;
        this.second = second;
    }
}
