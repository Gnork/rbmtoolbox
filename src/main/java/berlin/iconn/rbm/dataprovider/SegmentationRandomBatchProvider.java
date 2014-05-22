/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package berlin.iconn.rbm.dataprovider;

import java.util.Random;
import org.jblas.FloatMatrix;
import org.jblas.ranges.IntervalRange;

/**
 *
 * @author christoph
 */
public final class SegmentationRandomBatchProvider extends ATrainingDataProvider{
    
    private final Random random = new Random();
    private final float[][] images;
    private final int[][] labels;
    private final String[] classes;
    private final int edgeLength, batchXOffset, batchYOffset, batchCount;
    private final boolean isRGB;

    public SegmentationRandomBatchProvider(float[][] images, int edgeLength, int[][] labels, String[] classes, int batchXOffset, int batchYOffset, int batchCount, boolean isRGB) {
        super(new float[][]{{1}});
        this.images = images;
        this.labels = labels;
        this.classes = classes;
        this.batchXOffset = batchXOffset;
        this.batchYOffset = batchYOffset;
        this.batchCount = batchCount;
        this.isRGB = isRGB;
        this.edgeLength = edgeLength;
        changeDataAtTraining();
    }

    @Override
    public FloatMatrix getDataWithBiasForTraining() {
        return getDataWithBias();
    }

    @Override
    public FloatMatrix getTransposedDataWithBiasForTraining() {
        return getTransposedDataWithBias();
    }

    @Override
    public void changeDataAtTraining() {
        setDataWithBias(null);
        setTransData(null);
        setTransDataWithBias(null);
        int batchWidth = batchXOffset * 2 + 1;
        int batchHeight = batchYOffset * 2 + 1; 
        float[][] result = new float[batchCount][];
        for (int i = 0; i < batchCount; i++) {
            int indexImage = random.nextInt(images.length);
            int indexRow = random.nextInt(edgeLength - batchHeight);
            int indexColumn = random.nextInt(edgeLength - batchWidth);
            float[] newData;
            if(isRGB){
                newData = new float[batchWidth * batchHeight * 3];
                for(int j = 0; j < batchHeight; ++j){
                    for(int k = 0; k < batchWidth; ++k){
                        int batchPos = batchWidth * 3 * j + k * 3;
                        int imagePos = (indexRow + j) * edgeLength * 3 + (indexColumn + k) * 3;
                        newData[batchPos] = images[indexImage][imagePos];
                        newData[batchPos+1] = images[indexImage][imagePos+1];
                        newData[batchPos+2] = images[indexImage][imagePos+2];
                    }
                }
                
            }else{
                newData = new float[batchWidth * batchHeight];
                for(int j = 0; j < batchHeight; ++j){
                    for(int k = 0; k < batchWidth; ++k){
                        int batchPos = batchWidth * j + k;
                        int imagePos = (indexRow + j) * edgeLength + (indexColumn + k);
                        newData[batchPos] = images[indexImage][imagePos];
                    }
                }
            }
            int label = labels[indexImage][(indexColumn + batchXOffset) + (indexRow + batchYOffset) * edgeLength];
            float[] newLabels = new float[classes.length];
            newLabels[label] = 1f;
            result[i] = concatArrays(newLabels, newData);
        }
        FloatMatrix newBatches = new FloatMatrix(result);
        setData(newBatches);
        setMeanVector(FloatMatrix.zeros(batchCount, 1));
    }

    @Override
    public FloatMatrix getMeanVectorForTraining() {
        return getMeanVector();
    }
    
    public static float[] concatArrays(float[] ... arrays){
        int size = 0, count = 0;
        for(float[] a : arrays) size += a.length;
        float[] r = new float[size];
        for(float[] a : arrays) for(int i = 0; i < a.length; ++i, ++count) r[count] = a[i];
        return r;
    }
}
