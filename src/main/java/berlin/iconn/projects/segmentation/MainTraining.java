package berlin.iconn.projects.segmentation;

import berlin.iconn.persistence.InOutOperations;
import berlin.iconn.rbm.DataConverter;
import berlin.iconn.rbm.DataSet;
import berlin.iconn.rbm.RBM;
import berlin.iconn.rbm.StoppingCondition;
import berlin.iconn.rbm.WeightsFactory;
import berlin.iconn.rbm.dataprovider.SegmentationRandomBatchProvider;
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
public class MainTraining {

    private static final int edgeLength = 256;
    private static final int batchOffset = 2;
    private static final int padding = 0;
    private static final boolean isRGB = false;
    private static final boolean binarize = false;
    private static final boolean invert = false;
    private static final float minData = 0.0f;
    private static final float maxData = 1.0f;
    private static final String images = "Data/SiftFlowDataset_small/Images/spatial_envelope_256x256_static_8outdoorcategories";
    private static final String siftFlowLabels = "Data/SiftFlowDataset_small/SemanticLabels/labels";
    private static final String siftFlowClasses = "Data/SiftFlowDataset_small/SemanticLabels/classes.mat";
    
    public static void main(String[] args) {

        DataSet[] trainingDataSet;
        final String[] classes;
        final int[][] labels;
        
        try {
            trainingDataSet = InOutOperations.loadImages(new File(images), edgeLength, padding, binarize, invert, minData, maxData, isRGB);
            System.out.println("Images loaded");
            labels = InOutOperations.loadSiftFlowLabels(new File(siftFlowLabels));
            System.out.println("Labels loaded");
            classes = InOutOperations.loadSiftFlowClasses(new File(siftFlowClasses));
            System.out.println("Classes loaded");
        } catch (IOException ex) {
            Logger.getLogger(MainTraining.class.getName()).log(Level.SEVERE, null, ex);
            return;
        }
        
        final float[][] trainingData = DataConverter.dataSetToArray(trainingDataSet);
        
        int batchSize = 2 * batchOffset + 1;
        int inputSize = (batchSize * batchSize) + classes.length;

        RBMEnhancer enhancer = new RBMEnhancer(new RBM(WeightsFactory.randomGaussianWeightsWithBias(inputSize, inputSize / 2, 0.01f)));

        enhancer.addEnhancement(new WeightsSaver(10000));

        FloatMatrix[] batchSelectionData =  new FloatMatrix[trainingData.length];
        //prepare data for batch selection
        for (int i = 0; i < trainingData.length; i++) {
            batchSelectionData[i] = new FloatMatrix(edgeLength, edgeLength, trainingData[i]);
        }
        
        System.out.println("Start training");

        enhancer.train(new SegmentationRandomBatchProvider( batchSelectionData, labels, classes, batchOffset, batchOffset, 100),
                new StoppingCondition(1000000),
                new ConstantLearningRate(0.2f));
    }
}
