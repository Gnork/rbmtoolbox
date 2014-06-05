package berlin.iconn.rbm.dataprovider;

import org.jblas.FloatMatrix;
import org.jblas.ranges.IntervalRange;

import java.util.Random;

/**
 * Created by Moritz on 4/27/2014.
 */
public class RandomPictureBatchSelectionProvider extends ATrainingDataProvider {

    private final Random random = new Random();
    private final FloatMatrix[] pictures;
    private final int batchWidth, batchHeight, batchCount;

    public RandomPictureBatchSelectionProvider(FloatMatrix[] pictures, int batchCount, int batchWidth, int batchHeight, int seed) {
        super(new float[][]{{1}});
        this.pictures = pictures;
        this.batchWidth = batchWidth;
        this.batchHeight = batchHeight;
        this.batchCount = batchCount;
        random.setSeed(seed);
        changeDataAtTraining();
    }

    public RandomPictureBatchSelectionProvider(FloatMatrix[] pictures, int batchCount, int batchWidth, int batchHeight) {
        super(new float[][]{{1}});
        this.pictures = pictures;
        this.batchWidth = batchWidth;
        this.batchHeight = batchHeight;
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
        float[][] newData = new float[batchCount][];
        for (int i = 0; i < batchCount; i++) {
            FloatMatrix picture = pictures[random.nextInt(pictures.length)];
            int indexRow = random.nextInt(picture.getRows() - batchHeight);
            int indexColumn = random.nextInt(picture.getColumns() - batchWidth);
            newData[i] = picture.get(new IntervalRange(indexRow, indexRow + batchHeight),
                    new IntervalRange(indexColumn, indexColumn + batchWidth)).toArray();
        }
        FloatMatrix newBatches = new FloatMatrix(newData);
        FloatMatrix meanVector =  newBatches.rowMeans();
        setMeanVector(meanVector);
        setData(newBatches.subColumnVector(meanVector));
    }

    @Override
    public FloatMatrix getMeanVectorForTraining() {
        return getMeanVector();
    }
}
