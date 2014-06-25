package berlin.iconn.projects.segmentation.gnork;

import berlin.iconn.persistence.InOutOperations;
import berlin.iconn.rbm.StoppingCondition;
import berlin.iconn.rbm.learningRate.ConstantLearningRate;
import java.io.File;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Created by Moritz on 4/28/2014.
 */
public class MainTraining {

    private static final int edgeLength = 256;
    private static final int batchOffset = 2;
    private static final int padding = 0;
    private static final boolean isRGB = true;
    private static final boolean binarize = false;
    private static final boolean invert = false;
    private static final float minData = 0.0f;
    private static final float maxData = 1.0f;
    private static final String images = "Data/SiftFlowDataset/Images/spatial_envelope_256x256_static_8outdoorcategories";
    private static final String imagesCrossValidation = "Data/SiftFlowDataset/Images/spatial_envelope_256x256_static_8outdoorcategories";
    private static final String siftFlowLabels = "Data/SiftFlowDataset/SemanticLabels/labels";
    private static final String siftFlowLabelsCrossValidation = "Data/SiftFlowDataset/SemanticLabels/labels";
    private static final String siftFlowClasses = "Data/SiftFlowDataset/SemanticLabels/classes.mat";
    
    public static void main(String[] args) {

        final String[] classes;
        
        try {
            classes = InOutOperations.loadSiftFlowClasses(new File(siftFlowClasses));
            System.out.println("Classes loaded");
            
        } catch (IOException ex) {
            Logger.getLogger(MainTraining.class.getName()).log(Level.SEVERE, null, ex);
            return;
        }
        
        int batchSize = 2 * batchOffset + 1;
        int imageInputSize = (batchSize * batchSize);
        if(isRGB) imageInputSize *= 3;
        
        System.out.println("image input size: " + imageInputSize);
        System.out.println("label input size: " + classes.length);
        
        RandomSiftFlowLoader loader = new RandomSiftFlowLoader(new File(images), new File(siftFlowLabels), edgeLength, binarize, invert, minData, maxData, isRGB);
        RandomSiftFlowLoader loaderCrossValidation = new RandomSiftFlowLoader(new File(imagesCrossValidation), new File(siftFlowLabelsCrossValidation), edgeLength, binarize, invert, minData, maxData, isRGB);
        
        SegmentationStackRandomBatchGenerator provider = new SegmentationStackRandomBatchGenerator(loader, edgeLength, classes, batchOffset, batchOffset, 1000, 750, isRGB);
        SegmentationStackRandomBatchGenerator providerCrossValidation = new SegmentationStackRandomBatchGenerator(loaderCrossValidation, edgeLength, classes, batchOffset, batchOffset, 1000, 750, isRGB);
        
        RBMSegmentationStack stack = new RBMSegmentationStack(classes.length, 30, imageInputSize, 400, 100, 30, 0.01f, true);
        
        System.out.println("start training");
        
        stack.train(provider, providerCrossValidation, new StoppingCondition(100000), new ConstantLearningRate(0.2f));
    }
}
