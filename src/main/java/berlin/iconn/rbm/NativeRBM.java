package berlin.iconn.rbm;

import berlin.iconn.rbm.dataprovider.ATrainingDataProvider;
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




    @Override
    public void train(ATrainingDataProvider dataProvider, StoppingCondition stop, ILearningRate learningRate) {

        FloatMatrix data = dataProvider.getDataWithBiasForTraining();
        float[] mean = dataProvider.getMeanVector().data;
        createNativeRBM(
                weights.data, weights.getColumns(),
                data.data, data.rows, data.columns, mean,
                learningRate.getRate(), Runtime.getRuntime().availableProcessors());
 {
            float error = 1.0f;
            while(stop.isNotDone()) {
                learningRate.changeRate(error);
                if(binarize) {
                    trainNativeBinarizedWithError();
                } else {
                    trainNativeWithError();
                }
                dataProvider.changeDataAtTraining();
                error = getNativeError();
                stop.update(error);
            }
        }


    }

    @Override
    public float getError(float[][] data) {
        return 0;
    }

    @Override
    public float[][] getHidden(float[][] data) {
        return new float[0][];
    }

    @Override
    public float[][] getVisible(float[][] data) {
        return new float[0][];
    }

    @Override
    public float[][] getWeights() {
        return new float[0][];
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
    private native void setNativeWeights(float[] weights, int weightsCols);
    private native float[] getNativeWeights();
    private native float getNativeError();
    private native void setNativeData(float[] data, int dataRows, int dataCols, float[] mean);
}
