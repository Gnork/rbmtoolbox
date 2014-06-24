package berlin.iconn.rbm.dataprovider;

import org.jblas.FloatMatrix;

import java.util.Random;

/**
 * Created by Moritz on 4/27/2014.
 */
public class RandomBatchTrainingDataProvider extends ATrainingDataProvider {

    private static final Random RANDOM = new Random();
    private final FloatMatrix allData;
    private final int batchSize;

    public RandomBatchTrainingDataProvider(FloatMatrix data, int batchSize) {
        super(new float[0][]);
        this.batchSize = batchSize;
        allData = data;
        changeDataAtTraining();
    }


    @Override
    public void changeDataAtTraining() {
        setDataWithBias(null);
        float[][] minibatch = new float[batchSize][];
        float[][] data = allData.toArray2();
        for (int i = 0; i < batchSize; i++) {
            final int select = RANDOM.nextInt(data.length);
            minibatch[i] = data[select];
        }
        setData(new FloatMatrix(minibatch));
    }
}
