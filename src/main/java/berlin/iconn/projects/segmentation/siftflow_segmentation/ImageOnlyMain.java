package berlin.iconn.projects.segmentation.siftflow_segmentation;

import berlin.iconn.persistence.InOutOperations;
import berlin.iconn.rbm.*;
import berlin.iconn.rbm.dataprovider.ATrainingDataProvider;
import berlin.iconn.rbm.dataprovider.CascadeDataProvider;
import berlin.iconn.rbm.dataprovider.RandomPoolPatchDataProvider;
import berlin.iconn.rbm.enhancements.CrossValidator;
import berlin.iconn.rbm.enhancements.RBMEnhancer;
import berlin.iconn.rbm.enhancements.TrainingVisualizer;
import berlin.iconn.rbm.enhancements.visualizations.PatchVisualization;
import berlin.iconn.rbm.learningRate.ConstantLearningRate;
import org.jblas.FloatMatrix;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Date;

/**
 * Created by Moritz on 7/6/2014.
 */
public class ImageOnlyMain {

    private static final String IMAGES = "Data/SiftFlowDataset/Images/spatial_envelope_256x256_static_8outdoorcategories";
    private static final String CROSS = "Data/SiftFlowDataset/Images/crossvalidation_images";
    private static final int edgeLength = 256;
    private static final int patchSize = 16;
    private static final Date date = new Date();
    public static void main(String[] args) {

        File[] imageFiles = InOutOperations.getImageFiles(new File(IMAGES));
        File[] crossFiles = InOutOperations.getImageFiles(new File(CROSS));
        ATrainingDataProvider imageData = new RandomPoolPatchDataProvider(patchSize, 0.1f, edgeLength, imageFiles);
        //ATrainingDataProvider crossData = new RandomPoolPatchDataProvider(patchSize, 0.1f, edgeLength, crossFiles);
        int hidden1 = (patchSize * patchSize);
        float[][] weights = null;
        try {
            weights = InOutOperations.loadSimpleWeights(new File("Output/SimpleWeights/2014_07_13_13_25_56_RBM1_landscapeRGB_imageSize256_from_768_to_768.dat") );
          //  hidden1 = weights[0].length - 1;
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }

        int hidden2 = hidden1 / 2;
        //IRBM rbm1 = new NativeCudaRBM(weights);
        IRBM rbm1 = new NativeCudaRBM(WeightsFactory.randomGaussianWeightsWithBias(imageData.getData().getColumns(), hidden1, 0.01f));
        IRBM rbm2 = new NativeCudaRBM(WeightsFactory.randomGaussianWeightsWithBias(hidden1, hidden2, 0.01f));

        BufferedImage test = null;
        try {
            test = ImageIO.read(new File("Data/SiftFlowDataset/Images/cross.jpg"));
        } catch (IOException e) {
            e.printStackTrace();
        }

        PatchVisualization visualization = new PatchVisualization(patchSize, 1024, test, new IRBM[]{rbm1, rbm2});
        new Frame(visualization);

        RBMEnhancer enhancer = new RBMEnhancer(rbm1);
        enhancer.addEnhancement(new TrainingVisualizer(1, visualization));
        System.out.println("RBM: " + imageData.getData().getColumns() + " Hidden: " + hidden1);
        enhancer.train(imageData, new StoppingCondition(1_000_000), new ConstantLearningRate(0.01f));
        try {
            InOutOperations.saveSimpleWeights(enhancer.getWeights(), date, "RBM1_landscapeRGB_imageSize" + edgeLength + "_from_" + (patchSize * patchSize * 3) + "_to_" + hidden1);
        } catch (IOException e) {
            e.printStackTrace();
        }
//        visualization.nextState();
//        enhancer = new RBMEnhancer(rbm2);
//        enhancer.addEnhancement(new TrainingVisualizer(1, visualization));
//        System.out.println("RBM: " + hidden1 + " Hidden: " + hidden2);
//        enhancer.train(new CascadeDataProvider(rbm1, imageData), new StoppingCondition(1_000_000), new ConstantLearningRate(0.01f));
//        try {
//            InOutOperations.saveSimpleWeights(enhancer.getWeights(), date, "RBM2_landscapeRGB_imageSize" + edgeLength + "_from_" + hidden1 + "_to_" + hidden2);
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
    }
}
