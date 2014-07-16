package berlin.iconn.projects.segmentation.multiscale;

import berlin.iconn.persistence.InOutOperations;
import berlin.iconn.projects.segmentation.smallsegmentation.ShowSegmentationClasses;
import berlin.iconn.rbm.*;
import berlin.iconn.rbm.enhancements.RBMEnhancer;
import berlin.iconn.rbm.enhancements.TrainingVisualizer;
import berlin.iconn.rbm.learningRate.ConstantLearningRate;
import org.apache.commons.io.FilenameUtils;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

/**
 * Created by Moritz on 7/11/2014.
 */
public class MultiScaleMain {

    private static final String IMAGES = "Data/SiftFlowDataset/Images/spatial_envelope_256x256_static_8outdoorcategories";
    private static final String LABELS = "Data/SiftFlowDataset/SemanticLabels/labels";
    private static final String CLASSES = "Data/SiftFlowDataset/SemanticLabels/classes.mat";
    private static final Random RANDOM = new Random();
    public static void main(String[] args) {
        int classesCount = 33;
        try {
            String[] siftflowClasses = InOutOperations.loadSiftFlowClasses(new File(CLASSES));
        } catch (IOException e) {
            e.printStackTrace();
        }
        classesCount++;
        File[] imageFiles = InOutOperations.getImageFiles(new File(IMAGES));
        File[] labelFiles =  Arrays.asList(imageFiles).stream().map(file -> getLabelFile(file)).toArray(File[]::new);

        float[][] weights = null;
        try {
            weights = InOutOperations.loadSimpleWeights(new File("Output/SimpleWeights/2014_07_15_17_04_58_RBM1_landscapeRGB_imageSize256_from_768_to_256.dat") );
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }

        int[] imageSizes = {256, 128, 64, 16};
        IRBM[] imageIRBMS = new IRBM[]{new NativeRBM(weights)};
        MultiScaleProvider provider = new MultiScaleProvider(0.01f, imageFiles, labelFiles, imageIRBMS,imageSizes, classesCount);
        RBMEnhancer enhancer = new RBMEnhancer(new NativeRBM(WeightsFactory.randomGaussianWeightsWithBias(provider.getData().columns, 70, 0.01f)));

        int index = RANDOM.nextInt(imageFiles.length);
        ShowMultiScaleLabels show = new ShowMultiScaleLabels(provider.loadFromFiles(labelFiles[index], imageFiles[index]), provider, classesCount, new IRBM[]{ enhancer});
        new Frame(show);
        enhancer.addEnhancement(new TrainingVisualizer(1, show));

        enhancer.train(provider, new StoppingCondition(40_000), new ConstantLearningRate(0.01f));

    }


    private static File getLabelFile(File imageFile) {
        String labelName = FilenameUtils.removeExtension(imageFile.getName()) + ".mat";
        return new File(new File(LABELS).getAbsoluteFile() + "/" + labelName);
    }
}
