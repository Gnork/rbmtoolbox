package berlin.iconn.rbm;

import org.jblas.FloatMatrix;

import java.util.Random;

/**
 * Created by Moritz on 4/28/2014.
 */
public class WeightsFactory {
    private static final Random RANDOM = new Random();

    public static FloatMatrix randomGaussianWeightsWithBias(int input, int output, float weightScale, int seed) {
        FloatMatrix weights = new FloatMatrix(input, output);
        RANDOM.setSeed(seed);
        for (int i = 0; i < weights.data.length; i++) {
            weights.data[i] = (float) RANDOM.nextGaussian() * weightScale;
        }
        final FloatMatrix oneVectorCol = FloatMatrix.zeros(weights.getRows(), 1);
        final FloatMatrix oneVectorRow = FloatMatrix.zeros(1, weights.getColumns() + 1);

        weights = FloatMatrix.concatHorizontally(oneVectorCol, weights);
        weights = FloatMatrix.concatVertically(oneVectorRow, weights);
        return weights;
    }
    public static FloatMatrix randomGaussianWeightsWithBias(int input, int output, float weightScale) {
        FloatMatrix weights = new FloatMatrix(input, output);
        for (int i = 0; i < weights.data.length; i++) {
            weights.data[i] = (float) RANDOM.nextGaussian() * weightScale;
        }
        final FloatMatrix oneVectorCol = FloatMatrix.zeros(weights.getRows(), 1);
        final FloatMatrix oneVectorRow = FloatMatrix.zeros(1, weights.getColumns() + 1);

        weights = FloatMatrix.concatHorizontally(oneVectorCol, weights);
        weights = FloatMatrix.concatVertically(oneVectorRow, weights);
        return weights;
    }
    public static FloatMatrix patternWeightsWithBias(int input, int output, float weightScale) {
        FloatMatrix weights = new FloatMatrix(input, output);
        for (int i = 0; i < weights.data.length; i++) {
            weights.data[i] = ((i % 3 == 0) ? 1 : -1)  * weightScale;
        }
        final FloatMatrix oneVectorCol = FloatMatrix.zeros(weights.getRows(), 1);
        final FloatMatrix oneVectorRow = FloatMatrix.zeros(1, weights.getColumns() + 1);

        weights = FloatMatrix.concatHorizontally(oneVectorCol, weights);
        weights = FloatMatrix.concatVertically(oneVectorRow, weights);
        return weights;
    }
}
