package berlin.iconn.projects.segmentation.siftflow_segmentation;

import berlin.iconn.persistence.InOutOperations;
import berlin.iconn.rbm.*;
import berlin.iconn.rbm.dataprovider.ATrainingDataProvider;
import berlin.iconn.rbm.dataprovider.RandomPoolPatchDataProvider;
import berlin.iconn.rbm.enhancements.CrossValidator;
import berlin.iconn.rbm.enhancements.RBMEnhancer;
import berlin.iconn.rbm.learningRate.ConstantLearningRate;

import java.io.File;

/**
 * Created by Moritz on 7/6/2014.
 */
public class ImageOnlyMain {

    private static final String IMAGES = "Data/SiftFlowDataset/Images/spatial_envelope_256x256_static_8outdoorcategories";
    private static final String CROSS = "Data/SiftFlowDataset/Images/crossvalidation_images";
    private static final int edgeLength = 256;
    private static final int patchSize = 16;
    public static void main(String[] args) {

        File[] imageFiles = InOutOperations.getImageFiles(new File(IMAGES));
        File[] crossFiles = InOutOperations.getImageFiles(new File(CROSS));
        ATrainingDataProvider imageData = new RandomPoolPatchDataProvider(patchSize, 0.2f, edgeLength, imageFiles);
        ATrainingDataProvider crossData = new RandomPoolPatchDataProvider(patchSize, 0.1f, edgeLength, crossFiles);
        int hidden = 70;
        IRBM rbm = new NativeRBM(WeightsFactory.randomGaussianWeightsWithBias(imageData.getData().getColumns(), hidden, 0.01f));
        System.out.println("RBM: " + imageData.getData().getColumns() + " Hidden: " + hidden);
        RBMEnhancer enhancer = new RBMEnhancer(rbm);
        CrossValidator validator = new CrossValidator(rbm, crossData, 1000);
        enhancer.addEnhancement(validator);
        enhancer.train(imageData, new StoppingCondition(200_000), new ConstantLearningRate(0.01f));
    }
}
