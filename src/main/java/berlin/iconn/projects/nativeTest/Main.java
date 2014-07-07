package berlin.iconn.projects.nativeTest;

import berlin.iconn.matrixExperiments.PlaygroundRBM;
import berlin.iconn.persistence.InOutOperations;
import berlin.iconn.rbm.*;
import berlin.iconn.rbm.dataprovider.ATrainingDataProvider;
import berlin.iconn.rbm.dataprovider.BatchTrainingDataProvider;
import berlin.iconn.rbm.enhancements.RBMEnhancer;
import berlin.iconn.rbm.enhancements.TrainingVisualizer;
import berlin.iconn.rbm.enhancements.visualizations.FeatureDataVisualization;
import berlin.iconn.rbm.learningRate.ConstantLearningRate;
import org.jblas.FloatMatrix;

import java.io.File;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Created by Moritz on 5/29/2014.
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
    private static final float maxData = 0.5f;
    private static final String images = "Data/MNIST_1000";


    public static void main(String[] args) {
        DataSet[] trainingDataSet;
        try {
            trainingDataSet = InOutOperations.loadImages(new File(images), padding, binarize, invert, minData, maxData, isRGB);
        } catch (IOException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
            return;
        }
        final float[][] trainingData = DataConverter.dataSetToArray(trainingDataSet);

        final FloatMatrix weights = WeightsFactory.randomGaussianWeightsWithBias(edgeLength * edgeLength, 200, 0.01f);
        final StoppingCondition stoppingCondition = new StoppingCondition(20000);
        final ConstantLearningRate learningRate = new ConstantLearningRate(0.01f);
        final ATrainingDataProvider data = new BatchTrainingDataProvider(trainingData, 100);

        IRBM rbm = new RBM(weights);
        RBMEnhancer enhancer = new RBMEnhancer(rbm);
        FeatureDataVisualization featureDataVisualization = new FeatureDataVisualization(1, 30, edgeLength, trainingData);
        new Frame(featureDataVisualization);
        enhancer.addEnhancement(new TrainingVisualizer(1, featureDataVisualization));

        enhancer.train(data, stoppingCondition, learningRate);
    }
}
