/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package berlin.iconn.projects.segmentation;

import berlin.iconn.rbm.dataprovider.ATrainingDataProvider;
import java.util.Random;
import org.jblas.FloatMatrix;

/**
 *
 * @author christoph
 */
public final class SegmentationStackRandomBatchGenerator{
    private final Random random = new Random();
    private final RandomSiftFlowLoader loader;
    private final String[] classes;
    private final int edgeLength, batchXOffset, batchYOffset, batchCount;
    private final boolean isRGB;
    
    private FloatMatrix labelData;
    private FloatMatrix imageData;

    public SegmentationStackRandomBatchGenerator(RandomSiftFlowLoader loader, int edgeLength, String[] classes, int batchXOffset, int batchYOffset, int batchCount, boolean isRGB) {
        this.loader = loader;
        this.classes = classes;
        this.batchXOffset = batchXOffset;
        this.batchYOffset = batchYOffset;
        this.batchCount = batchCount;
        this.isRGB = isRGB;
        this.edgeLength = edgeLength;
        changeDataAtTraining();
    }   

    public void changeDataAtTraining() {
        int batchWidth = batchXOffset * 2 + 1;
        int batchHeight = batchYOffset * 2 + 1; 
        float[][] labelDataArray = new float[batchCount][];
        float[][] imageDataArray = new float[batchCount][];
        
        float[][] images = new float[batchCount][];
        int[][] labels = new int[batchCount][];
        
        loader.loadRandomImagesAndLabels(images, labels);
        
        for (int i = 0; i < batchCount; i++) {
            int indexRow = random.nextInt(edgeLength - batchHeight);
            int indexColumn = random.nextInt(edgeLength - batchWidth);
            if(isRGB){
                imageDataArray[i] = new float[batchWidth * batchHeight * 3];
                for(int j = 0; j < batchHeight; ++j){
                    for(int k = 0; k < batchWidth; ++k){
                        int batchPos = batchWidth * 3 * j + k * 3;
                        int imagePos = (indexRow + j) * edgeLength * 3 + (indexColumn + k) * 3;
                        imageDataArray[i][batchPos] = images[i][imagePos];
                        imageDataArray[i][batchPos+1] = images[i][imagePos+1];
                        imageDataArray[i][batchPos+2] = images[i][imagePos+2];
                    }
                }
                
            }else{
                imageDataArray[i] = new float[batchWidth * batchHeight];
                for(int j = 0; j < batchHeight; ++j){
                    for(int k = 0; k < batchWidth; ++k){
                        int batchPos = batchWidth * j + k;
                        int imagePos = (indexRow + j) * edgeLength + (indexColumn + k);
                        imageDataArray[i][batchPos] = images[i][imagePos];
                    }
                }
            }
            int label = labels[i][(indexColumn + batchXOffset) + (indexRow + batchYOffset) * edgeLength];
            labelDataArray[i] = new float[classes.length];
            labelDataArray[i][label] = 1f;
        }
        labelData = new FloatMatrix(labelDataArray);
        imageData = new FloatMatrix(imageDataArray);
    }
    
    public FloatMatrix getImageData(){
        return imageData;
    }
    
    public FloatMatrix getLabelData(){
        return labelData;
    }
}
