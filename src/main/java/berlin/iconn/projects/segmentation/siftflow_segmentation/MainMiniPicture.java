package berlin.iconn.projects.segmentation.siftflow_segmentation;

import berlin.iconn.persistence.InOutOperations;
import berlin.iconn.projects.segmentation.smallsegmentation.ShowSegmentation2;
import berlin.iconn.rbm.*;
import berlin.iconn.rbm.dataprovider.ATrainingDataProvider;
import berlin.iconn.rbm.enhancements.RBMEnhancer;
import berlin.iconn.rbm.enhancements.TrainingVisualizer;
import berlin.iconn.rbm.learningRate.ConstantLearningRate;
import org.apache.commons.io.FilenameUtils;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;

/**
 * Created by Moritz on 6/25/2014.
 */
public class MainMiniPicture {

    private static final Random RANDOM = new Random();

    private static final String IMAGES = "Data/SiftFlowDataset/Images/spatial_envelope_256x256_static_8outdoorcategories";
    private static final String LABELS = "Data/SiftFlowDataset/SemanticLabels/labels";
    private static final String CLASSES = "Data/SiftFlowDataset/SemanticLabels/classes.mat";
    public static void main(String[] args) throws IOException {

        int classesCount = 33;
        try {
            String[] siftflowClasses = InOutOperations.loadSiftFlowClasses(new File(CLASSES));
        } catch (IOException e) {
            e.printStackTrace();
        }
        classesCount++;
        File[] imageFiles = InOutOperations.getImageFiles(new File(IMAGES));
        File[] labelFiles =  Arrays.asList(imageFiles).stream().map(file -> getLabelFile(file)).toArray(File[]::new);

        int imageSize = 128;
        int miniImageSize = 16;
        int patchSize = 16;
        int output = 200;
        float loadNewFilePosibility = 0.05f;

        ATrainingDataProvider provider = new SiftFlowDataProviderWithMiniImage(classesCount, patchSize,loadNewFilePosibility,imageSize,miniImageSize, labelFiles, imageFiles);
        System.out.println("RBM1 == INPUT:  " + provider.getData().getColumns() + "  OUTPUT:  " + output);

        RBMEnhancer rbm = new RBMEnhancer(new NativeCudaRBM(WeightsFactory.randomGaussianWeightsWithBias(provider.getData().getColumns(), output, 0.01f)));

        int count = 4;
        ShowSegmentationPatchWiseWithMiniImage[] pics = new ShowSegmentationPatchWiseWithMiniImage[count];
        HashSet<Integer> picSelection = new HashSet<>();
        while(picSelection.size() < count) picSelection.add(RANDOM.nextInt(imageFiles.length));
        Integer[] selectionIndexs = picSelection.toArray(new Integer[0]);
        for (int i = 0; i < count; i++) {
            int pictureNumber = selectionIndexs[i];
            float[] image = InOutOperations.getImageData(imageSize, false, false, 0.0f, 1.0f, true, imageFiles[pictureNumber]).getData();
            float[] miniImage = InOutOperations.getImageData(miniImageSize, false, false, 0.0f, 1.0f, true, imageFiles[pictureNumber]).getData();
            int[] label = InOutOperations.loadSiftFlowLabel(imageSize, labelFiles[pictureNumber]);

            pics[i] = new ShowSegmentationPatchWiseWithMiniImage(label, image, miniImage, patchSize, classesCount, imageSize, imageSize, rbm);
            rbm.addEnhancement(new TrainingVisualizer(1, pics[i]));

        }
        new MultiFrame(pics);


        rbm.train(provider, new StoppingCondition(20_000), new ConstantLearningRate(0.01f));
    }


    private static File getLabelFile(File imageFile) {
        String labelName = FilenameUtils.removeExtension(imageFile.getName()) + ".mat";
        return new File(new File(LABELS).getAbsoluteFile() + "/" + labelName);
    }


}
