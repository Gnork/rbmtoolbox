package berlin.iconn.projects.bumblebee;

import berlin.iconn.persistence.InOutOperations;
import berlin.iconn.rbm.*;
import berlin.iconn.rbm.dataprovider.ATrainingDataProvider;
import berlin.iconn.rbm.dataprovider.BatchTrainingDataProvider;
import berlin.iconn.rbm.dataprovider.FullTrainingDataProvider;
import berlin.iconn.rbm.enhancements.RBMEnhancer;
import berlin.iconn.rbm.enhancements.TrainingVisualizer;
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
public class Main {

    private static final boolean exportImages = true;
    private static final String exportPath = "export";
    private static final int edgeLength = 28;
    private static final int padding = 0;
    private static final boolean isRGB = false;
    private static final boolean binarize = false;
    private static final boolean invert = false;
    private static final float minData = 0.0f;
    private static final float maxData = 1.0f;
    private static final String images = "Data\\_mnist20";


    public static void main(String[] args) {

        DataSet[] trainingDataSet;
        try {
            trainingDataSet = InOutOperations.loadImages(new File(images), edgeLength, padding, binarize, invert, minData, maxData, isRGB);
        } catch (IOException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
            return;
        }
        final float[][] trainingData = DataConverter.dataSetToArray(trainingDataSet);


        ExecutorService executor = Executors.newFixedThreadPool(8);
        final float[][][] parts4 = split(trainingData, edgeLength, edgeLength);

        final float[][][] parts = new float[16][][];

        final int edgelength2 = edgeLength / 2;
        for (int i = 0; i < parts4.length; i++) {
            float[][][] temp = split(parts4[i], edgelength2, edgelength2);
            int index = 4 * i;
            parts[index] = temp[0];
            parts[index + 1] = temp[1];
            parts[index + 2] = temp[2];
            parts[index + 3] = temp[3];
        }

        final FutureTask<RBM>[] rbmFutures = new FutureTask[parts.length];
        for (int i = 0; i < parts.length; i++) {
            final int index = i;
            final float[][] part = parts[i];
            FutureTask<RBM> future;
            final FloatMatrix weights = WeightsFactory.randomGaussianWeightsWithBias(part[0].length, 7, 0.01f);
            final StoppingCondition stoppingCondition = new StoppingCondition(100000);
            final ConstantLearningRate learningRate = new ConstantLearningRate(0.5f);
            final ATrainingDataProvider data = new FullTrainingDataProvider(part);
            if(i == 0) {
                future = new FutureTask<>(() -> {
                    RBM rbm = new RBM(weights);
                    RBMEnhancer enhancer = new RBMEnhancer(rbm);

                    FeatureDataVisualization visualization = new FeatureDataVisualization(4, 20, edgeLength / 4, part);
                    new Frame(visualization);

                    enhancer.addEnhancement(new TrainingVisualizer(1, visualization));
                    enhancer.train(data, stoppingCondition, learningRate);
                    return rbm;
                });
            } else future = new FutureTask<>(() -> {
                System.out.println("Start: " + index);
                RBM rbm = new RBM(weights);
                rbm.train(data, stoppingCondition, learningRate);
                System.out.println("End: " + index);
                return rbm;
            });
            rbmFutures[i] = future;
            executor.execute(future);
        }

        final RBM[] rbms = new RBM[rbmFutures.length];
        for (int i = 0; i < rbms.length; i++) {
            try {
                rbms[i] = rbmFutures[i].get();
            } catch (ExecutionException | InterruptedException e) {
                e.printStackTrace();
            }
        }


        float[][][] visibles = new float[rbms.length][][];
        for (int i = 0; i < rbms.length; i++) {
            float[][] hidden = rbms[i].getHidden(parts[i]);
            visibles[i] = rbms[i].getVisible(hidden);
        }
        float[][][] visibles4 = new float[rbms.length/ 4][][];

        for (int i = 0; i < visibles4.length; i++) {
            int index =  4 * i;

            visibles4[i] = combine(new float[][][]{
                    visibles[index],
                    visibles[index + 1],
                    visibles[index + 2],
                    visibles[index + 3]
            }, edgelength2, edgelength2);
        }

        try {

            final Date date = new Date();
            InOutOperations.exportAsImage(trainingData, date, "original");
            InOutOperations.exportAsImage(combine(visibles4, edgeLength, edgeLength), date, "recon");
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
