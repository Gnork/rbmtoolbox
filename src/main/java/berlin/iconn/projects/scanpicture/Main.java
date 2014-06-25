package berlin.iconn.projects.scanpicture;

import berlin.iconn.matrixExperiments.PlaygroundRBM;
import berlin.iconn.persistence.InOutOperations;
import berlin.iconn.rbm.*;
import berlin.iconn.rbm.dataprovider.BatchTrainingDataProvider;
import berlin.iconn.rbm.dataprovider.FullTrainingDataProvider;
import berlin.iconn.rbm.dataprovider.RandomPictureBatchSelectionProvider;
import berlin.iconn.rbm.enhancements.RBMEnhancer;
import berlin.iconn.rbm.enhancements.TrainingVisualizer;
import berlin.iconn.rbm.enhancements.visualizations.FeatureDataVisualization;
import berlin.iconn.rbm.learningRate.ConstantLearningRate;
import berlin.iconn.rbm.logistic.ILogistic;
import berlin.iconn.rbm.weightmodifier.GrowingModifier;
import org.jblas.FloatMatrix;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Created by Moritz on 4/28/2014.
 */
public class Main {

    private static final boolean exportImages = true;
    private static final String exportPath = "export";
    private static final int edgeLength = 512;
    private static final int padding = 0;
    private static final boolean isRGB = false;
    private static final boolean binarize = false;
    private static final boolean invert = false;
    private static final float minData = 0.0f;
    private static final float maxData = 1.0f;
    private static final String images = "Data/RGB_R02_0600x0600";
    public static void main(String[] args) {

        int rbmEdgeLength = 64;
        DataSet[] trainingDataSet;
        try {
            trainingDataSet = InOutOperations.loadImages(new File(images), edgeLength, binarize, invert, minData, maxData, isRGB);
        } catch (IOException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
            return;
        }
        final float[][] trainingData = DataConverter.dataSetToArray(trainingDataSet);

       // RBMEnhancer enhancer = new RBMEnhancer(new RBM(WeightsFactory.randomGaussianWeightsWithBias(rbmEdgeLength * rbmEdgeLength, rbmEdgeLength * 4, 0.01f)));
        RBMEnhancer enhancer = new RBMEnhancer(
                new CudaRBM(
                        WeightsFactory.randomGaussianWeightsWithBias(rbmEdgeLength * rbmEdgeLength, 1024, 0.01f)));
        ScanPicture picture = new ScanPicture(new FloatMatrix(edgeLength, edgeLength, trainingData[new Random().nextInt(trainingData.length)]), rbmEdgeLength);
        new Frame(picture);

        enhancer.addEnhancement(new TrainingVisualizer(1,picture));

        FloatMatrix[] batchSelectionData =  new FloatMatrix[trainingData.length];
        //prepare data for batch selection
        for (int i = 0; i < trainingData.length; i++) {
            batchSelectionData[i] = new FloatMatrix(edgeLength, edgeLength, trainingData[i]);
        }

        ArrayList<float[]> dataRows = new ArrayList<>();

        final int count = edgeLength / rbmEdgeLength;
        for (int i = 0; i < trainingData.length; i++) {
            for (int j = 0; j < count; j++) {
                for (int k = 0; k < count; k++) {
                    float[] patch = new float[rbmEdgeLength * rbmEdgeLength];
                    for (int g = 0; g < rbmEdgeLength; g++) {
                        for (int h = 0; h < rbmEdgeLength; h++) {
                            patch[g * rbmEdgeLength + h] = trainingData[i][(j * rbmEdgeLength + g) * edgeLength + k * rbmEdgeLength + h];
                        }
                    }
                    dataRows.add(patch);
                }
            }
        }
        FeatureDataVisualization  feature = new FeatureDataVisualization(1, 35, rbmEdgeLength, dataRows.toArray(new float[0][]));
        new Frame(feature);
        enhancer.addEnhancement(new TrainingVisualizer(1,feature));

        enhancer.train(new RandomPictureBatchSelectionProvider(batchSelectionData, 20, rbmEdgeLength, rbmEdgeLength),
                new StoppingCondition(2_000_000),
                new ConstantLearningRate(0.01f));
    }
}
