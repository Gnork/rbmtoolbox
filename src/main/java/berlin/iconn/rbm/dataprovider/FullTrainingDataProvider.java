package berlin.iconn.rbm.dataprovider;

import org.jblas.FloatMatrix;

/**
 * Created by Moritz on 4/27/2014.
 */
public class FullTrainingDataProvider extends ATrainingDataProvider {

    public FullTrainingDataProvider(FloatMatrix data, FloatMatrix mean) {
        super(data, mean);
    }

    public FullTrainingDataProvider(FloatMatrix data) {
        super(data);
    }

    public FullTrainingDataProvider(float[][] data) {
        super(data);
    }

    @Override
    public FloatMatrix getDataWithBiasForTraining() {
        return super.getDataWithBias();
    }

    @Override
    public FloatMatrix getTransposedDataWithBiasForTraining() {
        return super.getTransposedDataWithBias();
    }

    @Override
    public void changeDataAtTraining() {

    }

    @Override
    public FloatMatrix getMeanVectorForTraining() {
        return super.getMeanVector();
    }
}
