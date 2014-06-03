/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package berlin.iconn.projects.segmentation;

import berlin.iconn.persistence.InOutOperations;
import berlin.iconn.rbm.DataConverter;
import java.io.File;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;
import org.jblas.FloatMatrix;

/**
 *
 * @author christoph
 */
public final class SimpleStackGenerator{
    private final Random random = new Random();
    private final String[] classes;
    private final int edgeLength, batchXOffset, batchYOffset;
    
    private FloatMatrix labelData;
    private FloatMatrix imageData;
    
    private static final String imageFile = "Data/SiftFlowDataset/Images/spatial_envelope_256x256_static_8outdoorcategories/highway_urb713.jpg";
    private static final String labelFile = "Data/SiftFlowDataset/SemanticLabels/labels/highway_urb713.mat";

    public SimpleStackGenerator(int edgeLength, String[] classes, int batchXOffset, int batchYOffset) {
        this.classes = classes;
        this.batchXOffset = batchXOffset;
        this.batchYOffset = batchYOffset;
        this.edgeLength = edgeLength;
    }

    public void changeDataAtTraining() {
        int batchWidth = batchXOffset * 2 + 1;
        int batchHeight = batchYOffset * 2 + 1; 
        
        
        float[] img = null;
        int[] lbl = null;
        
        try {
            img = DataConverter.processPixelData(ImageIO.read(new File(imageFile)), edgeLength, false, false, 0, 1, true);
            lbl = InOutOperations.loadSiftFlowLabel(new File(labelFile));
        } catch (IOException ex) {
            Logger.getLogger(SimpleStackGenerator.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        List<float[]> imageList = new LinkedList<>();
        List<float[]> labelList = new LinkedList<>();
        
        
        for(int indexRow = 0; indexRow < edgeLength - batchHeight; indexRow+= 5){
            for(int indexColumn = 0; indexColumn < edgeLength - batchWidth; indexColumn += 5){
                
                int imgLabel = lbl[(indexColumn + batchXOffset) + (indexRow + batchYOffset) * edgeLength];
                
                if(imgLabel == 0){
                    continue;
                }
                
                float[] labelArray = new float[classes.length];
                labelArray[imgLabel-1] = 1f;
                
                float[] imgPatch = new float[batchWidth * batchHeight * 3];
                
                for(int j = 0; j < batchHeight; ++j){
                    for(int k = 0; k < batchWidth; ++k){
                        int batchPos = batchWidth * 3 * j + k * 3;
                        int imagePos = (indexRow + j) * edgeLength * 3 + (indexColumn + k) * 3;
                        imgPatch[batchPos] = img[imagePos];
                        imgPatch[batchPos+1] = img[imagePos+1];
                        imgPatch[batchPos+2] = img[imagePos+2];
                    }
                }
                
                imageList.add(imgPatch);
                labelList.add(labelArray);
            }
        }
        
        labelData = new FloatMatrix(labelList.toArray(new float[0][]));
        imageData = new FloatMatrix(imageList.toArray(new float[0][]));
    }
    
    public FloatMatrix getImageData(){
        return imageData;
    }
    
    public FloatMatrix getLabelData(){
        return labelData;
    }
}
