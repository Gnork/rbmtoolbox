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
public class SegmentationRandomBatchProvider extends ATrainingDataProvider{
    
    private final Random random = new Random();
    private final FloatMatrix[] images;
    private final int[][] labels;
    private final String[] labelNames;
    private final int batchXOffset, batchYOffset, batchCount;

    public SegmentationRandomBatchProvider(FloatMatrix[] images, int[][] labels, String[] labelNames, int batchXOffset, int batchYOffset, int batchCount) {
        super(new float[][]{{1}});
        this.images = images;
        this.labels = labels;
        this.labelNames = labelNames;
        this.batchXOffset = batchXOffset;
        this.batchYOffset = batchYOffset;
        this.batchCount = batchCount;
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
            FloatMatrix image = images[indexImage];
            int indexRow = random.nextInt(image.getRows() - batchHeight) + batchYOffset;
            int indexColumn = random.nextInt(image.getColumns() - batchWidth) + batchXOffset;
            float[] newData = image.get(new IntervalRange(indexRow - batchYOffset, indexRow + batchYOffset),
                    new IntervalRange(indexColumn - batchXOffset, indexColumn + batchXOffset)).toArray();
            int label = labels[indexImage][indexColumn + indexRow * image.getColumns()];
            float[] newLabels = new float[labelNames.length];
            newLabels[label] = 1f;
            result[i] = concatArrays(newLabels, newData);
        }
        FloatMatrix newBatches = new FloatMatrix(result);
        setData(newBatches);
    }

    @Override
    public FloatMatrix getMeanVectorForTraining() {
        return FloatMatrix.zeros(batchCount, 1);
    }
    
    private float[] concatArrays(float[] ... arrays){
        int size = 0, count = 0;
        for(float[] array : arrays) size += array.length;
        float[] r = new float[size];
        for(float[] a : arrays) for(int i = 0; i < a.length; ++i, ++count) r[count] = a[i];
        return r;
    }
}
