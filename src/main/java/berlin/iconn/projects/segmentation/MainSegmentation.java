/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package berlin.iconn.projects.segmentation;

import berlin.iconn.persistence.InOutOperations;
import berlin.iconn.rbm.DataConverter;
import berlin.iconn.rbm.RBM;
import berlin.iconn.rbm.WeightsFactory;
import java.io.File;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;
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
    private static final String imageFile = "Data/SiftFlowDataset_small/Images/spatial_envelope_256x256_static_8outdoorcategories/highway_urb713.jpg";
    private static final String labelFile = "Data/SiftFlowDataset_small/SemanticLabels/labels/highway_urb713.jpg";
    private static final String siftFlowClassesPath = "Data/SiftFlowDataset_small/SemanticLabels/classes.mat";
    private static final String weightsFile = "/home/christoph/git/rbmtoolbox/Output/SimpleWeights/weights_21_05_2014_14_36_29.dat";
    
    public static void main(String[] args) {
        float[][] weights;
        String[] classes;
        float[] image;
        int[] label;
        try {
            weights = InOutOperations.loadSimpleWeights(new File(weightsFile));
            classes = InOutOperations.loadSiftFlowClasses(new File(siftFlowClassesPath));
            image = DataConverter.processPixelData(ImageIO.read(new File(imageFile)), edgeLength, binarize, invert, minData, maxData, isRGB);
            label = InOutOperations.loadSiftFlowLabel(new File(labelFile));
        } catch (IOException | ClassNotFoundException ex) {
            Logger.getLogger(MainSegmentation.class.getName()).log(Level.SEVERE, null, ex);
            return;
        }
        RBM rbm = new RBM(new FloatMatrix(weights));
        
    }
}
