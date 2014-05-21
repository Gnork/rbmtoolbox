package berlin.iconn.rbm.dataprovider;

import org.jblas.FloatMatrix;

/**
 * Created by Moritz on 4/27/2014.
 */
public abstract class ATrainingDataProvider {

    private FloatMatrix data;
    private FloatMatrix transData;
    private FloatMatrix meanVector;
    private FloatMatrix dataWithBias;
    private FloatMatrix transDataWithBias;

    public ATrainingDataProvider(FloatMatrix data) {

       // setMeanVector(data.rowMeans());
        setMeanVector(FloatMatrix.zeros(data.getRows(), 1));
        this.setData(data.subColumnVector(meanVector));
    }

    public ATrainingDataProvider(float[][] data) {
        this(new FloatMatrix(data));
    }

    public FloatMatrix getMeanVector() {
        return meanVector;
    }

    public FloatMatrix getData() {
        return data;
    }

    public FloatMatrix getTransposedData() {
        if(transData == null) {
            setTransData(data.transpose());
        }
        return transData;
    }

    public FloatMatrix getDataWithBias() {
        if(dataWithBias == null) {
            setDataWithBias(putBiasOnData(this.data));
        }
        return dataWithBias;
    }

    public FloatMatrix getTransposedDataWithBias() {
        if(transDataWithBias == null) {
            setTransDataWithBias(getDataWithBias().transpose());
        }
        return transDataWithBias;
    }
    protected void setData(FloatMatrix data) {
        this.data = data;
    }

    protected void setTransData(FloatMatrix transData) {
        this.transData = transData;
    }

    protected void setMeanVector(FloatMatrix meanVector) {
        this.meanVector = meanVector;
    }

    protected void setDataWithBias(FloatMatrix dataWithBias) {
        this.dataWithBias = dataWithBias;
    }

    protected void setTransDataWithBias(FloatMatrix transDataWithBias) {
        this.transDataWithBias = transDataWithBias;
    }

    public abstract FloatMatrix getDataWithBiasForTraining();

    public abstract FloatMatrix getTransposedDataWithBiasForTraining();

    public abstract void changeDataAtTraining();

    public abstract FloatMatrix getMeanVectorForTraining();

    protected FloatMatrix putBiasOnData(FloatMatrix data) {
        return FloatMatrix.concatHorizontally(FloatMatrix.ones(data.getRows(), 1), data);
    }

}
