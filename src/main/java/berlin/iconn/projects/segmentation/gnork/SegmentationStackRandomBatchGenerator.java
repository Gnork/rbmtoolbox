/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package berlin.iconn.projects.segmentation.gnork;

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
    private final int edgeLength, batchXOffset, batchYOffset, batchCount, imagePoolSize;
    private final boolean isRGB;
    
    private FloatMatrix labelData;
    private FloatMatrix imageData;

    public SegmentationStackRandomBatchGenerator(RandomSiftFlowLoader loader, int edgeLength, String[] classes, int batchXOffset, int batchYOffset, int batchCount, int imagePoolSize, boolean isRGB) {
        this.loader = loader;
        this.classes = classes;
        this.batchXOffset = batchXOffset;
        this.batchYOffset = batchYOffset;
        this.batchCount = batchCount;
        this.isRGB = isRGB;
        this.edgeLength = edgeLength;
        this.imagePoolSize = imagePoolSize;
    }
    
    public int getBatchCount(){
        return batchCount;
    }

    public void changeDataAtTraining() {
        int batchWidth = batchXOffset * 2 + 1;
        int batchHeight = batchYOffset * 2 + 1; 
        float[][] labelDataArray = new float[batchCount][];
        float[][] imageDataArray = new float[batchCount][];
        
        float[][] images = new float[imagePoolSize][];
        int[][] labels = new int[imagePoolSize][];
        
        loader.loadRandomImagesAndLabels(images, labels);
        
        int imageCount = 0;
        
        for (int i = 0; i < batchCount; i++) {
            int imageIndex = 0;
            int indexRow = 0;
            int indexColumn = 0;
            int label = 0;
            while(label == 0){
                imageIndex = imageCount % imagePoolSize;
                indexRow = random.nextInt(edgeLength - batchHeight);
                indexColumn = random.nextInt(edgeLength - batchWidth);
                label = labels[imageIndex][(indexColumn + batchXOffset) + (indexRow + batchYOffset) * edgeLength];
                ++imageCount;
            }
            
            labelDataArray[i] = new float[classes.length];
            labelDataArray[i][label-1] = 1f;
            if(isRGB){
                imageDataArray[i] = new float[batchWidth * batchHeight * 3];
                for(int j = 0; j < batchHeight; ++j){
                    for(int k = 0; k < batchWidth; ++k){
                        int batchPos = batchWidth * 3 * j + k * 3;
                        int imagePos = (indexRow + j) * edgeLength * 3 + (indexColumn + k) * 3;
                        imageDataArray[i][batchPos] = images[imageIndex][imagePos];
                        imageDataArray[i][batchPos+1] = images[imageIndex][imagePos+1];
                        imageDataArray[i][batchPos+2] = images[imageIndex][imagePos+2];
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
