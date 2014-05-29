package berlin.iconn.projects.segmentation;

import berlin.iconn.persistence.InOutOperations;
import berlin.iconn.rbm.DataConverter;
import berlin.iconn.rbm.DataSet;
import berlin.iconn.rbm.RBM;
import berlin.iconn.rbm.StoppingCondition;
import berlin.iconn.rbm.WeightsFactory;
import berlin.iconn.rbm.enhancements.RBMEnhancer;
import berlin.iconn.rbm.enhancements.WeightsSaver;
import berlin.iconn.rbm.learningRate.ConstantLearningRate;
import java.io.File;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.jblas.FloatMatrix;

/**
 * Created by Moritz on 4/28/2014.
 */
public class MainTrainingOld {

    private static final int edgeLength = 256;
    private static final int batchOffset = 2;
    private static final int padding = 0;
    private static final boolean isRGB = true;
    private static final boolean binarize = false;
    private static final boolean invert = false;
    private static final float minData = 0.0f;
    private static final float maxData = 1.0f;
    private static final String images = "Data/SiftFlowDataset_small/Images/spatial_envelope_256x256_static_8outdoorcategories";
    private static final String siftFlowLabels = "Data/SiftFlowDataset_small/SemanticLabels/labels";
    private static final String siftFlowClasses = "Data/SiftFlowDataset_small/SemanticLabels/classes.mat";
    private static final String weightsFile = "Output/SimpleWeights/weights_22_05_2014_01_02_32.dat";
    
    public static void main(String[] args) {

        DataSet[] trainingDataSet;
        final String[] classes;
        final int[][] labels;
        float[][] weights;
        
        try {
            trainingDataSet = InOutOperations.loadImages(new File(images), edgeLength, padding, binarize, invert, minData, maxData, isRGB);
            System.out.println("Images loaded");
            labels = InOutOperations.loadSiftFlowLabels(new File(siftFlowLabels));
            System.out.println("Labels loaded");
            classes = InOutOperations.loadSiftFlowClasses(new File(siftFlowClasses));
            System.out.println("Classes loaded");
            weights = InOutOperations.loadSimpleWeights(new File(weightsFile));
        } catch (IOException | ClassNotFoundException ex) {
            Logger.getLogger(MainTrainingOld.class.getName()).log(Level.SEVERE, null, ex);
            return;
        }
        
        final float[][] trainingData = DataConverter.dataSetToArray(trainingDataSet);
        
        if(labels.length != trainingData.length){
            System.err.println("number of pics and labels does not match");
            return;
        }
        
        int batchSize = 2 * batchOffset + 1;
        int inputSize = (batchSize * batchSize);
        if(isRGB) inputSize *= 3;
        inputSize += classes.length;
        
        System.out.println("input size: " + inputSize);
        
        FloatMatrix rbmWeights = new FloatMatrix(weights);
        // FloatMatrix rbmWeights = new FloatMatrix(WeightsFactory.randomGaussianWeightsWithBias(inputSize, inputSize / 2, 0.01f));
        RBMEnhancer enhancer = new RBMEnhancer(new RBM(rbmWeights));

        enhancer.addEnhancement(new WeightsSaver(10000));
        
        System.out.println("Start training");

        enhancer.train(new SegmentationRandomBatchProvider( trainingData, edgeLength, labels, classes, batchOffset, batchOffset, 100, isRGB),
                new StoppingCondition(1000000),
                new ConstantLearningRate(0.2f));
    }
}
