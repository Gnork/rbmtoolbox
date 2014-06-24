package berlin.iconn.rbm.dataprovider;

import org.jblas.FloatMatrix;

/**
 * Created by Moritz on 4/27/2014.
 */
public abstract class ATrainingDataProvider {

    private FloatMatrix data;
    private FloatMatrix dataWithBias;

    public ATrainingDataProvider(FloatMatrix data) {
        this.setData(data);
    }
    public ATrainingDataProvider(FloatMatrix data, FloatMatrix meanVector) {

        this.setData(data.subColumnVector(meanVector));
    }

    public ATrainingDataProvider(float[][] data) {
        this(new FloatMatrix(data));
    }


    public FloatMatrix getData() {
        return data;
    }

    public FloatMatrix getDataWithBias() {
        if(dataWithBias == null) {
            setDataWithBias(putBiasOnData(this.data));
        }
        return dataWithBias;
    }
    protected void setData(FloatMatrix data) {
        this.data = data;
    }

    protected void setDataWithBias(FloatMatrix dataWithBias) {
        this.dataWithBias = dataWithBias;
    }

    public abstract void changeDataAtTraining();

    protected FloatMatrix putBiasOnData(FloatMatrix data) {
        return FloatMatrix.concatHorizontally(FloatMatrix.ones(data.getRows(), 1), data);
    }

}
