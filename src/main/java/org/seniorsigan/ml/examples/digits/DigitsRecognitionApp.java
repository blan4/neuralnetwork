package org.seniorsigan.ml.examples.digits;

import com.google.gson.Gson;
import org.seniorsigan.ml.neuralnetwork.Network;
import org.seniorsigan.ml.neuralnetwork.NetworkData;
import org.seniorsigan.ml.neuralnetwork.TrainPair;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.List;

public class DigitsRecognitionApp {
    private final static String trainDataPath = "digits/train-images-idx3-ubyte.gz";
    private final static String trainLabelsPath = "digits/train-labels-idx1-ubyte.gz";
    private final static String testDataPath = "digits/t10k-images-idx3-ubyte.gz";
    private final static String testLabelsPath = "digits/t10k-labels-idx1-ubyte.gz";

    private final static Gson gson = new Gson();
    private final static ImageService imageService = new ImageService();

    public static void main(String[] args) throws IOException, URISyntaxException {
        train();
    }

    private static void train() throws URISyntaxException, IOException {
        MNISTLoader loader = new MNISTLoader();
        List<TrainPair> trainData = loader.load(res(trainDataPath), res(trainLabelsPath));
        List<TrainPair> testData = loader.load(res(testDataPath), res(testLabelsPath));
        //imageService.toImage(trainData.get(0).input.data);
        Network n = new Network(NetworkData.build(new int[]{28*28, 100, 10}), (got, expected) -> {
            //System.out.println("Got :"+ vectorToInt(got) + "; Expected: " + vectorToInt(expected));
            return vectorToInt(got) == vectorToInt(expected);
        });
        n.train(trainData.subList(0,50000), 30, 100, 3.0, trainData.subList(50000, 60000));
        NetworkData nd = n.getNetworkData();
        FileWriter fw = new FileWriter("digits_nd.data");
        gson.toJson(nd, fw);
    }

    private static int vectorToInt(double[] src) {
        int v = 0;
        double maximum = src[v];
        for (int i = 1; i < src.length; i++) {
            if (maximum < src[i]) {
                maximum = src[i];
                v = i;
            }
        }
        return v;
    }

    private static File res(String path) throws URISyntaxException {
        return new File(DigitsRecognitionApp.class.getClassLoader().getResource(path).toURI());
    }
}
