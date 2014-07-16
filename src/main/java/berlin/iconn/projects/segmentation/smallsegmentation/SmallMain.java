package berlin.iconn.projects.segmentation.smallsegmentation;

import berlin.iconn.persistence.InOutOperations;
import berlin.iconn.projects.segmentation.SegmentationDataConverter;
import berlin.iconn.rbm.*;
import berlin.iconn.rbm.Frame;
import berlin.iconn.rbm.dataprovider.ATrainingDataProvider;
import berlin.iconn.rbm.dataprovider.BatchTrainingDataProvider;
import berlin.iconn.rbm.dataprovider.RandomBatchTrainingDataProvider;
import berlin.iconn.rbm.enhancements.RBMEnhancer;
import berlin.iconn.rbm.enhancements.TrainingVisualizer;
import berlin.iconn.rbm.learningRate.ConstantLearningRate;
import org.jblas.FloatMatrix;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.event.WindowEvent;
import java.io.File;
import java.io.IOException;
import java.util.Date;

/**
 * Created by Moritz on 6/4/2014.
 */
public class SmallMain {

    private static final String siftFlowClasses = "Data/SiftFlowDataset/SemanticLabels/classes.mat";
    private static final String imageFile = "Data/SiftFlowDataset/Images/spatial_envelope_256x256_static_8outdoorcategories/highway_urb713.jpg";
    private static final String labelFile = "Data/SiftFlowDataset/SemanticLabels/labels/highway_urb713.mat";
    private static final int classLength = 33;
    private static final int pictureSize = 128;

    public static void main(String[] args) {
        final String[] classes;
        int[] labels;
        final float[] image;
        final Date date = new Date();

        try {
            classes = InOutOperations.loadSiftFlowClasses(new File(siftFlowClasses));
            System.out.println("Classes loaded");
            labels = InOutOperations.loadSiftFlowLabel(pictureSize, labelFile);

            image = DataConverter.processPixelData(ImageIO.read(new File(imageFile)), pictureSize, false, false, 0, 1, true);
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }


        //Arrays.asList(classes).stream().forEach(System.out::println);
        int patchSize = 16;
        FloatMatrix[] data = SegmentationDataConverter.createTrainingData(labels, image, pictureSize, patchSize, classLength);

        final FloatMatrix labelMatrix = data[0];
        final FloatMatrix imagePatchMatrix = data[1];

        segmentation2(labels, image, date, patchSize, labelMatrix, imagePatchMatrix);

    }

    private static void segmentation2(int[] labels, float[] image, Date date, int patchSize, FloatMatrix labelMatrix, FloatMatrix imagePatchMatrix) {
        FloatMatrix combinationMatrix = FloatMatrix.concatHorizontally(labelMatrix, imagePatchMatrix);


            int hiddenCombinationCount = 70;
            int hiddenAssociationCount = 33;
            IRBM combinationRBM = new NativeCudaRBM(WeightsFactory.randomGaussianWeightsWithBias(combinationMatrix.getColumns(), hiddenCombinationCount, 0.01f));
            IRBM associationRBM = new NativeCudaRBM(WeightsFactory.randomGaussianWeightsWithBias(hiddenCombinationCount, hiddenAssociationCount, 0.01f));
            ShowSegmentation2 showSegmentation = new ShowSegmentation2(labels, image, patchSize, classLength, pictureSize, pictureSize, combinationRBM, associationRBM);
            ShowSegmentationClasses showClasses = new ShowSegmentationClasses(labels, image, patchSize, classLength, pictureSize, pictureSize, combinationRBM, associationRBM);
            Frame frame = new Frame(showSegmentation);
            Frame frame2 = new Frame(showClasses);
            ATrainingDataProvider combinationData = new RandomBatchTrainingDataProvider(combinationMatrix, 10);
            System.out.println("Train Combination: " +  combinationRBM.getWeights().length + "  " + combinationRBM.getWeights()[0].length);
            RBMEnhancer enhancer = new RBMEnhancer(combinationRBM);
            enhancer.addEnhancement(new TrainingVisualizer(1, showSegmentation));
            enhancer.addEnhancement(new TrainingVisualizer(1, showClasses));
            enhancer.train(combinationData, new StoppingCondition(75_000), new ConstantLearningRate(0.01f));
            try {
                InOutOperations.saveSimpleWeights(combinationRBM.getWeights(), date, "combination");
            } catch (IOException e) {
                e.printStackTrace();
            }


            System.out.println("Error: " + showSegmentation.getSmallestZeroError() + " at Epoch: " + showSegmentation.getSmallestZeroErrorEpoch());
            System.out.println("MSE: " + showSegmentation.getSmallestMSE() + " at Epoch: " + showSegmentation.getSmallestZeroMSEEpoch());





//        showSegmentation.nextState();
//        FloatMatrix associationMatrix = new FloatMatrix(combinationRBM.getHidden(combinationMatrix.toArray2()));
//        ATrainingDataProvider associatioData = new RandomBatchTrainingDataProvider(associationMatrix, 10);
//        System.out.println("Train Association: " +  associationRBM.getWeights().length + "  " + associationRBM.getWeights()[0].length);
//        enhancer = new RBMEnhancer(associationRBM);
//        enhancer.addEnhancement(new TrainingVisualizer(1, showSegmentation));
//        enhancer.train(associatioData, new StoppingCondition(20_000), new ConstantLearningRate(0.01f));
//        try {
//            InOutOperations.saveSimpleWeights(associationRBM.getWeights(), date, "asssociation");
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
    }

    private static void segmentation1(int[] labels, float[] image, Date date, int patchSize, FloatMatrix labelMatrix, FloatMatrix imagePatchMatrix) {
        IRBM rbmLabel = new RBM(WeightsFactory.randomGaussianWeightsWithBias(classLength, classLength, 0.01f));
        IRBM rbmImage = new CudaRBM(WeightsFactory.randomGaussianWeightsWithBias(imagePatchMatrix.columns, 400, 0.01f));
        IRBM rbmCombination = new CudaRBM(WeightsFactory.randomGaussianWeightsWithBias(rbmLabel.getWeights()[0].length + rbmImage.getWeights()[0].length - 2, 400, 0.01f));
        IRBM rbmAssociation = new CudaRBM(WeightsFactory.randomGaussianWeightsWithBias(rbmCombination.getWeights()[0].length - 1, 200, 0.01f));
        ShowSegmentation1 visu = new ShowSegmentation1(labels, image, pictureSize, pictureSize,
                rbmLabel,
                rbmImage,
                rbmCombination,
                rbmAssociation,
                classLength, patchSize);
        new Frame(visu);

        ATrainingDataProvider labelData = new BatchTrainingDataProvider(labelMatrix, 15);
        System.out.println("Train Labels: " +  rbmLabel.getWeights().length + "  " + rbmLabel.getWeights()[0].length);
        RBMEnhancer enhancer = new RBMEnhancer(rbmLabel);
        enhancer.addEnhancement(new TrainingVisualizer(1, visu));
        enhancer.train(labelData, new StoppingCondition(300_000), new ConstantLearningRate(0.01f));
        try {
            InOutOperations.saveSimpleWeights(rbmLabel.getWeights(), date, "label");
        } catch (IOException e) {
            e.printStackTrace();
        }


        ATrainingDataProvider imageData = new BatchTrainingDataProvider(imagePatchMatrix, 15);
        System.out.println("Train Imagepatches: " +  rbmImage.getWeights().length + "  " + rbmImage.getWeights()[0].length);
        enhancer = new RBMEnhancer(rbmImage);
        enhancer.addEnhancement(new TrainingVisualizer(1, visu));
        enhancer.train(imageData, new StoppingCondition(300_000), new ConstantLearningRate(0.01f));
        try {
            InOutOperations.saveSimpleWeights(rbmImage.getWeights(), date, "image");
        } catch (IOException e) {
            e.printStackTrace();
        }

        float[][] hiddenLabels = rbmLabel.getHidden(labelMatrix.toArray2());
        float[][] hiddenImagePatches = rbmImage.getHidden(imagePatchMatrix.toArray2());
        FloatMatrix combinationMatrix = FloatMatrix.concatHorizontally(new FloatMatrix(hiddenLabels), new FloatMatrix(hiddenImagePatches));
        ATrainingDataProvider combinationData = new BatchTrainingDataProvider(combinationMatrix, 15);

        System.out.println("Train Combination: " +  rbmCombination.getWeights().length + "  " + rbmCombination.getWeights()[0].length);
        enhancer = new RBMEnhancer(rbmCombination);
        enhancer.addEnhancement(new TrainingVisualizer(1, visu));
        visu.nextState();
        enhancer.train(combinationData, new StoppingCondition(300_000), new ConstantLearningRate(0.01f));
        try {
            InOutOperations.saveSimpleWeights(rbmCombination.getWeights(), date, "combi");
        } catch (IOException e) {
            e.printStackTrace();
        }


        float[][] hiddenCombination = rbmCombination.getHidden(combinationMatrix.toArray2());
        FloatMatrix associationMatrix = new FloatMatrix(hiddenCombination);
        ATrainingDataProvider associationData = new BatchTrainingDataProvider(associationMatrix, 15);

        System.out.println("Train Association: " +  rbmAssociation.getWeights().length + "  " + rbmAssociation.getWeights()[0].length);
        enhancer = new RBMEnhancer(rbmAssociation);
        enhancer.addEnhancement(new TrainingVisualizer(1, visu));
        visu.nextState();
        enhancer.train(associationData, new StoppingCondition(300_000), new ConstantLearningRate(0.01f));
        try {
            InOutOperations.saveSimpleWeights(rbmCombination.getWeights(), date, "assoc");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // labels on FloatMatrix[0]
    // image patches om FloatMatrix[1]



    public static void show(float[] data, int rows, int columns, String name) {
        System.out.println(name);
        for (int i = 0; i < columns; i++) {
            for (int j = 0; j < rows; j++) {
                System.out.print(String.format("%.5f", data[i * rows + j]) + ", ");
            }
            System.out.println();
        }
        System.out.println();
    }

    public static void show(int[] data, int rows, int columns, String name) {
        System.out.println(name);
        for (int i = 0; i < columns; i++) {
            for (int j = 0; j < rows; j++) {
                System.out.print(data[i * rows + j] + ", ");
            }
            System.out.println();
        }
        System.out.println();
    }
}
