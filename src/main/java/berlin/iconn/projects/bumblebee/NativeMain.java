package berlin.iconn.projects.bumblebee;

import berlin.iconn.persistence.InOutOperations;
import berlin.iconn.rbm.*;
import berlin.iconn.rbm.dataprovider.ATrainingDataProvider;
import berlin.iconn.rbm.dataprovider.BatchTrainingDataProvider;
import berlin.iconn.rbm.dataprovider.FullTrainingDataProvider;
import berlin.iconn.rbm.enhancements.RBMEnhancer;
import berlin.iconn.rbm.enhancements.TrainingVisualizer;

import berlin.iconn.rbm.enhancements.visualizations.ErrorDataVisualization;
import berlin.iconn.rbm.enhancements.visualizations.FeatureDataVisualization;
import berlin.iconn.rbm.learningRate.ConstantLearningRate;
import org.jblas.FloatMatrix;

import java.io.File;
import java.io.IOException;
import java.util.Date;
import java.util.concurrent.*;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Created by Moritz on 5/18/2014.
 */
public class NativeMain {

    private static final boolean exportImages = true;
    private static final String exportPath = "export";
    private static final int edgeLength = 64;
    private static final int padding = 0;
    private static final boolean isRGB = false;
    private static final boolean binarize = false;
    private static final boolean invert = false;
    private static final float minData = 0.0f;
    private static final float maxData = 1.0f;
    private static final String images = "Data\\RGB_R02_0600x0600";


    public static void main(String[] args) {

        DataSet[] trainingDataSet;
        try {
            trainingDataSet = InOutOperations.loadImages(new File(images), edgeLength, padding, binarize, invert, minData, maxData, isRGB);
        } catch (IOException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
            return;
        }
        final float[][] trainingData = DataConverter.dataSetToArray(trainingDataSet);
        final FullTrainingDataProvider getMean = new FullTrainingDataProvider(new FloatMatrix(trainingData));
        final float[][][] parts = split(trainingData, edgeLength, edgeLength);




        final NativeRBM[] rbms = new NativeRBM[parts.length];

        for (int i = 0; i < parts.length; i++) {

            float[][] part = parts[i];
            final ConstantLearningRate learningRate = new ConstantLearningRate(0.01f);
            final FullTrainingDataProvider data = new FullTrainingDataProvider(new FloatMatrix(part));
            final FloatMatrix weights = WeightsFactory.randomGaussianWeightsWithBias(part[0].length, 100, 0.01f, 1337);
            System.out.println("Start: " + i);
            NativeRBM rbm = new NativeRBM(weights);
            rbm.fastTrain(data, 30000, learningRate);
            System.out.println("End: " + i);
            rbms[i] = rbm;

        }


        float[][][] visibles = new float[rbms.length][][];
        for (int i = 0; i < rbms.length; i++) {
            float[][] hidden = rbms[i].getHidden(parts[i]);
            visibles[i] = rbms[i].getVisible(hidden);
        }

        try {

            final Date date = new Date();
            InOutOperations.exportAsImage(trainingData, date, "original");
            InOutOperations.exportAsImage(combine(visibles, edgeLength, edgeLength), date, "recon");
//            InOutOperations.exportAsImage(featurePics, date, "features");
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    public static float[][][] split(float[][] src, int width, int height) {

        float[][][] dst = new float[4][src.length][];
        for (int i = 0; i < src.length; i++) {

            float[][] splitted = split(src[i], width, height);

            dst[0][i] = splitted[0];
            dst[1][i] = splitted[1];
            dst[2][i] = splitted[2];
            dst[3][i] = splitted[3];
        }

        return dst;
    }

    public static float[][] split(float[] src, int width, int height) {

        final int halfWidth = width / 2;
        final int halfHeight = height / 2;
        float[] dst1 = new float[halfWidth * halfHeight];
        float[] dst2 = new float[halfWidth * halfHeight];
        float[] dst3 = new float[halfWidth * halfHeight];
        float[] dst4 = new float[halfWidth * halfHeight];

        for (int y = 0; y < halfHeight; y++) {
            int sy1 = 2 * y;
            int sy2 = 2 * y + 1;
            int posY = halfWidth * y;
            for (int x = 0; x < halfWidth; x++) {
                int sx1 = 2 * x;
                int sx2 = 2 * x + 1;
                int pos = posY + x;
                dst1[pos] = src[sy1 * width + sx1];
                dst2[pos] = src[sy1 * width + sx2];
                dst3[pos] = src[sy2 * width + sx1];
                dst4[pos] = src[sy2 * width + sx2];
            }
        }

        return new float[][]{dst1, dst2, dst3, dst4};
    }


    public static float[][] combine(float[][][] src, int width, int height) {
        float[][] dst = new float[src[0].length][];

        for (int i = 0; i < src[0].length; i++) {
            dst[i] = combine(new float[][]{
                    src[0][i],
                    src[1][i],
                    src[2][i],
                    src[3][i]
            }, width, height);
        }

        return dst;
    }

    public static float[] combine(float[][] src, int width, int height) {
        float[] dst = new float[src[0].length * 4];

        final int halfWidth = width / 2;
        final int halfHeight = height / 2;
        for (int y = 0; y < halfHeight; y++) {
            int sy1 = 2 * y;
            int sy2 = 2 * y + 1;
            int posY = halfWidth * y;
            for (int x = 0; x < halfWidth; x++) {
                int sx1 = 2 * x;
                int sx2 = 2 * x + 1;
                int pos = posY + x;
                dst[sy1 * width + sx1] = src[0][pos];
                dst[sy1 * width + sx2] = src[1][pos];
                dst[sy2 * width + sx1] = src[2][pos];
                dst[sy2 * width + sx2] = src[3][pos];
            }
        }

        return dst;
    }
}
