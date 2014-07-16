package berlin.iconn.projects.scanpicture;

import berlin.iconn.matrixExperiments.PlaygroundRBM;
import berlin.iconn.persistence.InOutOperations;
import berlin.iconn.rbm.*;
import berlin.iconn.rbm.dataprovider.*;
import berlin.iconn.rbm.enhancements.RBMEnhancer;
import berlin.iconn.rbm.enhancements.TrainingVisualizer;
import berlin.iconn.rbm.enhancements.visualizations.FeatureDataVisualization;
import berlin.iconn.rbm.enhancements.visualizations.PatchVisualization;
import berlin.iconn.rbm.learningRate.ConstantLearningRate;
import berlin.iconn.rbm.logistic.ILogistic;
import berlin.iconn.rbm.weightmodifier.GrowingModifier;
import org.jblas.FloatMatrix;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Created by Moritz on 4/28/2014.
 */
public class MainScanPicture {

    private static final int edgeLength = 512;
    private static final String images = "Data\\SiftFlowDataset\\Images\\spatial_envelope_256x256_static_8outdoorcategories";
    public static void main(String[] args) {
        int patchLength = 128;
        RBMEnhancer enhancer = new RBMEnhancer(
                new NativeRBM(
                        WeightsFactory.randomGaussianWeightsWithBias(patchLength * patchLength * 3, 2500, 0.01f)));
        BufferedImage bf = new BufferedImage(edgeLength, edgeLength, BufferedImage.TYPE_INT_RGB);

        try {
            bf = ImageIO.read(new File("Data\\Pictures\\test3.jpg"));
        } catch (IOException e) {
            e.printStackTrace();
        }

        PatchVisualization picture = new PatchVisualization(patchLength, edgeLength, bf,new IRBM[]{enhancer});
        new Frame(picture);

        enhancer.addEnhancement(new TrainingVisualizer(1,picture));

        File[] imageFiles = InOutOperations.getImageFiles(images);

        enhancer.train(new RandomPoolPatchDataProvider(patchLength, 0.01f, edgeLength, imageFiles),
                new StoppingCondition(3_000_000),
                new ConstantLearningRate(0.01f));
    }
}
