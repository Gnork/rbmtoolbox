package berlin.iconn.rbm.dataprovider;

import org.jblas.FloatMatrix;

import java.util.Random;

/**
 * Created by Moritz on 7/9/2014.
 */
public class NoiseProvider extends ATrainingDataProvider {

    private final int size;
    private final Random random;

    public NoiseProvider(int size) {
        super(new float[1][1]);
        this.size = size;
        this.random = new Random();
        changeDataAtTraining();
    }

    @Override
    public void changeDataAtTraining() {
        setDataWithBias(null);
        float[][] data = getFloats();
        setData(new FloatMatrix(data));
    }

    private float[][] getFloats() {
        float[][] data = new float[10][size];

        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < size; j++) {
                data[i][j] = random.nextFloat();
            }
        }
        return data;
    }
}
