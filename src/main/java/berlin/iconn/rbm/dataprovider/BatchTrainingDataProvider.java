package berlin.iconn.rbm.dataprovider;

import org.jblas.FloatMatrix;

/**
 * Created by Moritz on 4/27/2014.
 */
public class BatchTrainingDataProvider extends ATrainingDataProvider {

    private final FloatMatrix allData;
    private int index = 0;
    private final int batchSize;

    public BatchTrainingDataProvider(FloatMatrix data, int batchSize) {
        super(new float[0][0]);
        changeDataAtTraining();
        allData = data;
        this.batchSize = batchSize;
    }

    public BatchTrainingDataProvider(float[][] data, int batchSize){
        this(new FloatMatrix(data), batchSize);
    }

    @Override
    public void changeDataAtTraining() {
        setDataWithBias(null);
        float[][] minibatch = new float[batchSize][];
        float[][] data = allData.toArray2();
        for (int i = 0; i < batchSize; i++) {
            minibatch[i] = data[index];
            index++;
            index %= allData.getRows();
        }
        setData(new FloatMatrix(minibatch));
    }
}
