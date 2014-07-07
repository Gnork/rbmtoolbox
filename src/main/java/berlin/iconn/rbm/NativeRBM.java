package berlin.iconn.rbm;

import berlin.iconn.rbm.dataprovider.ATrainingDataProvider;
import berlin.iconn.rbm.dataprovider.FullTrainingDataProvider;
import berlin.iconn.rbm.learningRate.ILearningRate;
import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;

/**
 * Created by Moritz on 5/27/2014.
 */
public class NativeRBM  implements IRBM, AutoCloseable {
    static {
        System.loadLibrary("CNativeRBM");
    }

    protected FloatMatrix weights;
    private boolean binarize;
    private final int id;


    public NativeRBM(FloatMatrix weights) {
        this(weights, false);
    }

    public NativeRBM(FloatMatrix weights, boolean binarize) {
        this.weights = weights;
        this.binarize = binarize;
        this.id =  createNativeRBM(weights.data, weights.rows, weights.columns, Runtime.getRuntime().availableProcessors());
    }

    @Override
    public void train(ATrainingDataProvider dataProvider, StoppingCondition stop, ILearningRate learningRate) {

        FloatMatrix data = dataProvider.getDataWithBias();
        setNativeData(id, data.data, data.rows);
        setNativeWeights(id, weights.data, weights.rows, weights.columns);

        float error = 1.0f;
        while (stop.isNotDone()) {
            learningRate.changeRate(error);
            setNativeLearningRate(id, learningRate.getRate());

            if (binarize) {
                trainNativeBinarizedWithError(id);
            } else {
                trainNativeWithError(id);
            }
            dataProvider.changeDataAtTraining();

            data = dataProvider.getDataWithBias();
            setNativeData(id, data.toArray(), data.rows);

            error = getNativeError(id);
            stop.update(error);

        }
        weights.data = getNativeWeights(id);
    }

    public void fastTrain(FullTrainingDataProvider dataProvider, int epochs, ILearningRate learningRate) {
        FloatMatrix data = dataProvider.getDataWithBias();
        setNativeData(id, data.data, data.rows);
        setNativeWeights(id, weights.data, weights.rows, weights.columns);
        if (binarize) {
            for (int i = 0; i < epochs; i++) {
                setNativeLearningRate(id, learningRate.getRate());
                trainNativeBinarizedWithoutError(id);
            }
        } else {
            for (int i = 0; i < epochs; i++) {
                setNativeLearningRate(id, learningRate.getRate());
                trainNativeWithoutError(id);
            }
        }
        weights.data = getNativeWeights(id);
    }


    public FloatMatrix getHidden(FloatMatrix data) {

        setNativeWeights(id, weights.data, weights.rows, weights.columns);
        FullTrainingDataProvider provider = new FullTrainingDataProvider(data);
        final FloatMatrix dataWithBias = provider.getDataWithBias();
        FloatMatrix hidden = FloatMatrix.zeros(dataWithBias.getRows(), weights.columns);
        hidden.data = runHidden(id, dataWithBias.data, dataWithBias.rows);
        return removeBiasFromData(hidden);
    }

    public FloatMatrix getVisible(FloatMatrix data) {

        setNativeWeights(id, weights.data, weights.rows, weights.columns);
        FullTrainingDataProvider provider = new FullTrainingDataProvider(data);
        final FloatMatrix dataWithBias = provider.getDataWithBias();
        FloatMatrix visible = FloatMatrix.zeros(dataWithBias.getRows(), weights.rows);
        visible.data = runVisible(id, dataWithBias.data, dataWithBias.rows);
        return removeBiasFromData(visible);
    }

    @Override
    public float getError(float[][] data) {
        return getError(new FullTrainingDataProvider(new FloatMatrix(data)));
    }

    public float getError(ATrainingDataProvider data) {
        final FloatMatrix dataWithBias = data.getDataWithBias();
        return error(id, dataWithBias.data, dataWithBias.rows);
    }

    @Override
    public float[][] getHidden(float[][] data) {
        return getHidden(new FloatMatrix(data)).toArray2();
    }

    @Override
    public float[][] getVisible(float[][] data) {
        return getVisible(new FloatMatrix(data)).toArray2();
    }

    @Override
    public float[][] getWeights() {
        return weights.toArray2();
    }

    private FloatMatrix removeBiasFromData(FloatMatrix data) {
        return data.getRange(0, data.getRows(), 1, data.getColumns());
    }


    @Override
    public void close() throws Exception {
        deleteNativeRBM(id);
        System.out.println("free Native RBM: " + id);
    }

    private native int createNativeRBM(
            float[] weights, int weightRows, int weightsCols,
            int threads);

    private native void deleteNativeRBM(int id);

    private native void trainNativeWithoutError(int id);

    private native void trainNativeWithError(int id);

    private native void trainNativeBinarizedWithoutError(int id);

    private native void trainNativeBinarizedWithError(int id);

    private native float[] getNativeWeights(int id);

    private native float getNativeError(int id);

    private native void setNativeWeights(int id, float[] weights,  int weightsRows, int weightsCols);

    private native void setNativeData(int id, float[] data, int dataRows);

    private native void setNativeLearningRate(int id, float learningRate);

    private native float[] runHidden(int id, float[] visible, int rows);
    private native float[] runVisible(int id, float[] hidden, int rows);
    private native float error(int id, float[] visible, int rows);

}
