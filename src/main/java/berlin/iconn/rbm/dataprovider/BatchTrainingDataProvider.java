package berlin.iconn.rbm.dataprovider;

import org.jblas.FloatMatrix;

/**
 * Created by Moritz on 4/27/2014.
 */
public class BatchTrainingDataProvider extends ATrainingDataProvider {

    private final FloatMatrix[] dataWithBiasBatches;
    private final FloatMatrix[] transDataWithBiasBatches;
    private final FloatMatrix[] meanVectors;
    private int batchIndex = 0;

    public BatchTrainingDataProvider(FloatMatrix data, int batchCount) {
        super(data);

        dataWithBiasBatches = new FloatMatrix[batchCount];
        transDataWithBiasBatches = new FloatMatrix[batchCount];
        meanVectors = new FloatMatrix[batchCount];

        final int range = data.getRows() / batchCount;
        final int threshold = data.getRows() % batchCount;
        for (int i = 0; i < batchCount; i++) {
            final int start = i * range + (i < threshold ? i : threshold);
            final int end = start + range + (i < threshold ? 1 : 0);
            dataWithBiasBatches[i] = putBiasOnData(getData().getRange(start, end, 0, getData().columns));
            transDataWithBiasBatches[i] = dataWithBiasBatches[i].transpose();
            meanVectors[i] = getMeanVector().getRange(start, end, 0, getMeanVector().columns);
        }
    }

    public BatchTrainingDataProvider(float[][] data, int batchSize){
        this(new FloatMatrix(data), batchSize);
    }

    @Override
    public FloatMatrix getDataWithBiasForTraining() {
        return dataWithBiasBatches[batchIndex];
    }

    @Override
    public FloatMatrix getTransposedDataWithBiasForTraining() {
        return transDataWithBiasBatches[batchIndex];
    }

    @Override
    public FloatMatrix getMeanVectorForTraining() {
        return meanVectors[batchIndex];
    }

    @Override
    public void changeDataAtTraining() {
        batchIndex++;
        if(batchIndex >= meanVectors.length) batchIndex = 0;
    }
}
