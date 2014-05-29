package berlin.iconn.projects.scanpicture;

import berlin.iconn.persistence.InOutOperations;
import berlin.iconn.rbm.*;
import berlin.iconn.rbm.dataprovider.RandomPictureBatchSelectionProvider;
import berlin.iconn.rbm.enhancements.RBMEnhancer;
import berlin.iconn.rbm.enhancements.TrainingVisualizer;
import berlin.iconn.rbm.learningRate.ConstantLearningRate;
import berlin.iconn.rbm.logistic.ILogistic;
import berlin.iconn.rbm.weightmodifier.GrowingModifier;
import org.jblas.FloatMatrix;
import java.io.File;
import java.io.IOException;
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
    private static final String images = "Data/Pictures";
    public static void main(String[] args) {

        int rbmEdgeLength = 8;
        DataSet[] trainingDataSet;
        try {
            trainingDataSet = InOutOperations.loadImages(new File(images), edgeLength, padding, binarize, invert, minData, maxData, isRGB);
        } catch (IOException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
            return;
        }
        final float[][] trainingData = DataConverter.dataSetToArray(trainingDataSet);

       // RBMEnhancer enhancer = new RBMEnhancer(new RBM(WeightsFactory.randomGaussianWeightsWithBias(rbmEdgeLength * rbmEdgeLength, rbmEdgeLength * 4, 0.01f)));
        RBMEnhancer enhancer = new RBMEnhancer(new NativeRBM(WeightsFactory.randomGaussianWeightsWithBias(rbmEdgeLength * rbmEdgeLength, rbmEdgeLength * rbmEdgeLength , 0.01f),false));
        ScanPicture picture = new ScanPicture(new FloatMatrix(edgeLength, edgeLength, trainingData[0]), rbmEdgeLength);
        new Frame(picture);

        enhancer.addEnhancement(new TrainingVisualizer(1,picture));

        FloatMatrix[] batchSelectionData =  new FloatMatrix[trainingData.length];
        //prepare data for batch selection
        for (int i = 0; i < trainingData.length; i++) {
            batchSelectionData[i] = new FloatMatrix(edgeLength, edgeLength, trainingData[i]);
        }

        enhancer.train(new RandomPictureBatchSelectionProvider( batchSelectionData, 100, rbmEdgeLength, rbmEdgeLength ),
                new StoppingCondition(1000000),
                new ConstantLearningRate(0.01f));
    }
}
