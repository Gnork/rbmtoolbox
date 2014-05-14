package berlin.iconn.rbm.dataprovider;

import org.jblas.FloatMatrix;
import org.jblas.ranges.IntervalRange;

import java.util.Random;

/**
 * Created by Moritz on 4/27/2014.
 */
public class FilterPictureBatchProvider extends ATrainingDataProvider {
    private final FloatMatrix[] pictures;
    private final int batchWidth, batchHeight, batchCount;

    private int indexRow = 0, indexColumn = 0, indexPicture = 0;

    public FilterPictureBatchProvider(FloatMatrix[] pictures, int batchCount, int batchWidth, int batchHeight) {
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

            FloatMatrix picture = pictures[indexPicture];
            newData[i] = picture.get(new IntervalRange(indexRow, indexRow + batchHeight),
                    new IntervalRange(indexColumn, indexColumn + batchWidth)).toArray();

            indexColumn++;
            if(indexColumn >= picture.getColumns() - batchWidth) {
                indexColumn = 0;
                indexRow++;
                if(indexRow >= picture.getRows() - batchHeight) {
                    indexRow = 0;
                    indexPicture++;
                    if(indexPicture >= pictures.length) {
                        indexPicture = 0;
                    }
                }
            }
        }

        FloatMatrix newBatches = new FloatMatrix(newData);
        FloatMatrix meanVector = newBatches.rowMeans();
        setMeanVector(meanVector);
        setData(newBatches.subColumnVector(meanVector));
    }

    @Override
    public FloatMatrix getMeanVectorForTraining() {
        return getMeanVector();
    }
}
