/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package berlin.iconn.projects.segmentation;

import berlin.iconn.persistence.InOutOperations;
import berlin.iconn.rbm.DataConverter;
import berlin.iconn.rbm.Frame;
import java.io.File;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;

import berlin.iconn.rbm.enhancements.TrainingVisualizer;
import berlin.iconn.rbm.enhancements.visualizations.ErrorDataVisualization;
import org.jblas.FloatMatrix;

/**
 *
 * @author christoph
 */
public class MainSegmentation {
    private static final int edgeLength = 256;
    private static final int batchOffset = 2;
    private static final int padding = 0;
    private static final boolean isRGB = true;
    private static final boolean binarize = false;
    private static final boolean invert = false;
    private static final float minData = 0.0f;
    private static final float maxData = 1.0f;
    private static final String imageFile = "Data/SiftFlowDataset/Images/spatial_envelope_256x256_static_8outdoorcategories/highway_urb713.jpg";
    private static final String labelFile = "Data/SiftFlowDataset/SemanticLabels/labels/highway_urb713.mat";
    private static final String siftFlowClassesPath = "Data/SiftFlowDataset/SemanticLabels/classes.mat";
    private static final String dateString = "2014_05_28_14_36_57";
    
    public static void main(String[] args) {
        String imageWeightsFile = "Output/SimpleWeights/" + dateString + "_image.dat";
        String labelWeightsFile = "Output/SimpleWeights/" + dateString + "_label.dat";
        String combiWeightsFile = "Output/SimpleWeights/" + dateString + "_combi.dat";
        String assocWeightsFile = "Output/SimpleWeights/" + dateString + "_assoc.dat";
        
        float[][] labelWeights;
        float[][] imageWeights;
        float[][] combiWeights;
        float[][] assocWeights;
        
        String[] classes;
        float[] image;
        int[] label;
        
        try {
            labelWeights = InOutOperations.loadSimpleWeights(new File(labelWeightsFile));
            imageWeights = InOutOperations.loadSimpleWeights(new File(imageWeightsFile));
            combiWeights = InOutOperations.loadSimpleWeights(new File(combiWeightsFile));
            assocWeights = InOutOperations.loadSimpleWeights(new File(assocWeightsFile));
            classes = InOutOperations.loadSiftFlowClasses(new File(siftFlowClassesPath));
            image = DataConverter.processPixelData(ImageIO.read(new File(imageFile)), edgeLength, binarize, invert, minData, maxData, isRGB);
            label = InOutOperations.loadSiftFlowLabel(new File(labelFile));
        } catch (IOException | ClassNotFoundException ex) {
            Logger.getLogger(MainSegmentation.class.getName()).log(Level.SEVERE, null, ex);
            return;
        }
        RBMSegmentationStack stack = new RBMSegmentationStack(new FloatMatrix(labelWeights),
                new FloatMatrix(imageWeights), new FloatMatrix(combiWeights),
                new FloatMatrix(assocWeights), false);

        StackVisualization vis = new StackVisualization(stack, image, label, classes, minData, isRGB, batchOffset);
        new Frame(vis);
        

        // OriginalLabelVisualisation vis2 = new OriginalLabelVisualisation(stack, label, classes, minData, isRGB, batchOffset);
        // new Frame(vis2);

    }
}
