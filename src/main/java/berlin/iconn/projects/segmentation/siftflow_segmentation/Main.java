package berlin.iconn.projects.segmentation.siftflow_segmentation;

import berlin.iconn.persistence.InOutOperations;
import berlin.iconn.projects.segmentation.smallsegmentation.AShowSegmentation;
import berlin.iconn.rbm.*;
import berlin.iconn.rbm.dataprovider.ATrainingDataProvider;
import berlin.iconn.rbm.enhancements.RBMEnhancer;
import berlin.iconn.rbm.enhancements.TrainingVisualizer;
import berlin.iconn.rbm.learningRate.ConstantLearningRate;
import org.apache.commons.io.FilenameUtils;

import javax.swing.*;
import java.awt.*;
import java.awt.event.WindowEvent;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;

/**
 * Created by Moritz on 6/25/2014.
 */
public class Main {

    private static final Random RANDOM = new Random(0);

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
        int patchSize = 16;

        float loadNewFilePosibility = 0.05f;
        for (int j = 7; j < 40 ; j++) {
            int output = 10 * j;
            RANDOM.setSeed(3);
            ATrainingDataProvider provider = new SiftFlowDataProvider(imageFiles, labelFiles,imageSize, patchSize, classesCount, loadNewFilePosibility);
            System.out.println("RBM1 == INPUT:  " + (provider.getData().getColumns() - 1) + "  OUTPUT:  " + output);

            RBMEnhancer rbm = new RBMEnhancer(new NativeRBM(WeightsFactory.randomGaussianWeightsWithBias(provider.getData().getColumns(), output, 0.01f)));

            int count = 12;
            ShowSegmentationPatchWise[] pics = new ShowSegmentationPatchWise[count];
            HashSet<Integer> picSelection = new HashSet<>();
            while(picSelection.size() < count) picSelection.add(RANDOM.nextInt(imageFiles.length));
            Integer[] selectionIndexs = picSelection.toArray(new Integer[0]);
            for (int i = 0; i < count; i++) {
                int pictureNumber = selectionIndexs[i];
                float[] image = InOutOperations.getImageData(imageSize, false, false, 0.0f, 1.0f, true, imageFiles[pictureNumber]).getData();
                int[] label = InOutOperations.loadSiftFlowLabel(imageSize, labelFiles[pictureNumber]);
                pics[i] = new ShowSegmentationPatchWise(label, image, patchSize, classesCount, imageSize, imageSize, rbm);
                rbm.addEnhancement(new TrainingVisualizer(1, pics[i]));

            }
            MultiFrame frame = new MultiFrame(pics);
            rbm.train(provider, new StoppingCondition(100_000), new ConstantLearningRate(0.01f));
            float sumError = 0;
            float sumEpoch = 0;
            float sumMSE = 0;
            float sumMSEEpoch = 0;
            for(AShowSegmentation showSegmentation: pics) {
                sumError += showSegmentation.getSmallestZeroError();
                sumEpoch += showSegmentation.getSmallestZeroErrorEpoch();
                sumMSE += showSegmentation.getSmallestMSE();
                sumMSEEpoch += showSegmentation.getSmallestZeroMSEEpoch();
            }

            System.out.println("Error:\t" + sumError / count);
            System.out.println("Epoch:\t" + sumEpoch / count);
            System.out.println("MSE:  \t" + sumMSE / count);
            System.out.println("Epoch:\t" + sumMSEEpoch / count);
            System.out.println();

            frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
            WindowEvent windowEvent = new WindowEvent(frame, WindowEvent.WINDOW_CLOSING);
            Toolkit.getDefaultToolkit().getSystemEventQueue().postEvent(windowEvent);
        }
    }


    private static File getLabelFile(File imageFile) {
        String labelName = FilenameUtils.removeExtension(imageFile.getName()) + ".mat";
        return new File(new File(LABELS).getAbsoluteFile() + "/" + labelName);
    }


}
