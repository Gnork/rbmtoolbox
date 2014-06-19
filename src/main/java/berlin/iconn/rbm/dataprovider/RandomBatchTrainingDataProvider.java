package berlin.iconn.rbm.dataprovider;

import org.jblas.FloatMatrix;

import java.util.Random;

/**
 * Created by Moritz on 4/27/2014.
 */
public class RandomBatchTrainingDataProvider extends ATrainingDataProvider {

    private static final Random RANDOM = new Random();
    private FloatMatrix dataWithBiasSelection;
    private FloatMatrix transDataWithBiasBatches;
    private final int batchSize;

    public RandomBatchTrainingDataProvider(FloatMatrix data, int batchSize) {
        super(data);
        this.batchSize = batchSize;
        changeDataAtTraining();
    }

    @Override
    public FloatMatrix getDataWithBiasForTraining() {
        return dataWithBiasSelection;
    }

    @Override
    public FloatMatrix getTransposedDataWithBiasForTraining() {
        return transDataWithBiasBatches;
    }

    @Override
    public void changeDataAtTraining() {
        float[][] selection =  new float[batchSize][];
        float[][] data = getDataWithBias().toArray2();
        for (int i = 0; i < batchSize; i++) {
            final int select = RANDOM.nextInt(data.length);
            selection[i] = data[select];
        }

        dataWithBiasSelection = new FloatMatrix(selection);
        transDataWithBiasBatches = new FloatMatrix(selection).transpose();
    }
}
