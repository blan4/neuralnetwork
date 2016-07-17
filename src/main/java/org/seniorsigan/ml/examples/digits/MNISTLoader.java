package org.seniorsigan.ml.examples.digits;

import org.seniorsigan.ml.neuralnetwork.TrainPair;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.GZIPInputStream;

public class MNISTLoader {
    public List<TrainPair> load(File dataPath, File labelsPath) throws IOException {
        GZIPInputStream dataStream = new GZIPInputStream(new FileInputStream(dataPath));
        GZIPInputStream labelsStream = new GZIPInputStream(new FileInputStream(labelsPath));
        dataStream.skip(4); // skip magic number
        byte[] buf = new byte[4];
        dataStream.read(buf);
        int numberOfImages = ByteBuffer.wrap(buf).getInt();
        labelsStream.skip(8);
        dataStream.skip(8);
        List<TrainPair> trainPairs = new ArrayList<>(numberOfImages);
        for (int i = 0; i < numberOfImages; i++) {
            byte[] imageBuf = new byte[28*28];
            double[] label = vectorizeLabel(labelsStream.read());
            dataStream.read(imageBuf);
            double[] image = normalize(imageBuf);
            trainPairs.add(TrainPair.build(image, label));
        }
        return trainPairs;
    }

    private double[] normalize(byte[] src) {
        double[] out = new double[src.length];
        for (int i = 0; i < src.length; ++i) {
            out[i] = Byte.toUnsignedInt(src[i]) / 255.0;
        }
        return out;
    }

    private double[] vectorizeLabel(int l) {
        double[] v = new double[10];
        v[l] = 1.0;
        return v;
    }
}
