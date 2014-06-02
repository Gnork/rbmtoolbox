package berlin.iconn.rbm;

import berlin.iconn.rbm.dataprovider.ATrainingDataProvider;
import berlin.iconn.rbm.dataprovider.FullTrainingDataProvider;
import berlin.iconn.rbm.learningRate.ILearningRate;
import org.jblas.FloatMatrix;

/**
 * Created by Moritz on 5/27/2014.
 */
public class NativeRBM implements IRBM {
    static {
        System.loadLibrary("CNativeRBM");
    }
    private FloatMatrix weights;
    private boolean binarize;


    private final GetStatesFunction getHiddenFunction;
    private final GetStatesFunction getVisibleFunction;

    public NativeRBM(FloatMatrix weights) {
        this(weights, false);
    }

    public NativeRBM(FloatMatrix weights, boolean binarize) {
        this.weights = weights;
        this.binarize = binarize;

        this.getVisibleFunction = this.getHiddenFunction = new GetStatesFunction();
    }

    @Override
    public void train(ATrainingDataProvider dataProvider, StoppingCondition stop, ILearningRate learningRate) {

        FloatMatrix data = dataProvider.getDataWithBiasForTraining();
        float[] mean = dataProvider.getMeanVector().toArray();
        createNativeRBM(
            weights.toArray(), weights.getColumns(),
            data.toArray(), data.rows, data.columns, mean,
            learningRate.getRate(), Runtime.getRuntime().availableProcessors());
        {
            float error = 1.0f;
            while(stop.isNotDone()) {
                learningRate.changeRate(error);
                setNativeLearningRate(learningRate.getRate());

                if(binarize) {
                    trainNativeBinarizedWithError();
                } else {
                    trainNativeWithError();
                }
                dataProvider.changeDataAtTraining();

                data = dataProvider.getDataWithBiasForTraining();
                mean = dataProvider.getMeanVector().data;
                setNativeData(data.toArray(), data.rows, data.columns, mean);

                error = getNativeError();
                stop.update(error);
            }
        }

        weights.data = getNativeWeights();
        deleteNativeRBM();

    }
    
    public void fastTrain(FullTrainingDataProvider dataProvider, int epochs, ILearningRate learningRate)
    {
        FloatMatrix data = dataProvider.getDataWithBiasForTraining();
        float[] mean = dataProvider.getMeanVector().data;
        createNativeRBM(
                weights.data, weights.getColumns(),
                data.data, data.rows, data.columns, mean,
                learningRate.getRate(), Runtime.getRuntime().availableProcessors());
        if(binarize) {
            for (int i = 0; i < epochs; i++) {
                setNativeLearningRate(learningRate.getRate());
                trainNativeBinarizedWithoutError();
            }
        } else {
            for (int i = 0; i < epochs; i++) {
                setNativeLearningRate(learningRate.getRate());
                trainNativeWithoutError();
            }
        }

         weights.data = getNativeWeights();

        deleteNativeRBM();
    }


    public FloatMatrix getHidden(FloatMatrix data) {
        FullTrainingDataProvider provider = new FullTrainingDataProvider(data);
        return removeBiasFromData(getHiddenFunction.get(provider.getDataWithBias(), weights));
    }


    public FloatMatrix getVisible(FloatMatrix data) {
        FullTrainingDataProvider provider = new FullTrainingDataProvider(data);
        return removeBiasFromData(getHiddenFunction.get(provider.getDataWithBias(), weights.transpose()));
    }

    @Override
    public float getError(float[][] data) {
        return 0;
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

    private native void createNativeRBM(
            float[] weights, int weightsCols,
            float[] data, int dataRows, int dataCols,
            float[] mean,
            float learningRate,
            int threads);

    private native void deleteNativeRBM();

    private native void trainNativeWithoutError();
    private native void trainNativeWithError();
    private native void trainNativeBinarizedWithoutError();
    private native void trainNativeBinarizedWithError();
    private native float[] getNativeWeights();
    private native float getNativeError();
    private native void setNativeWeights(float[] weights, int weightsCols);
    private native void setNativeData(float[] data, int dataRows, int dataCols, float[] mean);
    private native void setNativeLearningRate(float learningRate);
}
