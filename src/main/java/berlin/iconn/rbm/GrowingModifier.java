package berlin.iconn.rbm;

import berlin.iconn.rbm.ITrainingModifier;
import org.jblas.FloatMatrix;

/**
 * Created by Moritz on 4/30/2014.
 */
public class GrowingModifier implements ITrainingModifier {
    private float lastError = 1.0f;
    private float t =  1f;

    @Override
    public FloatMatrix modify(FloatMatrix weights, float error, int currentEpochs) {
        lastError = lastError * (1.0f - t) + t * error;
        weights.subi(weights.mul( (float) Math.pow(currentEpochs + 2, - 0.95) ));
        if(lastError - error < 0.01f) {
            System.out.println(lastError - error);
            lastError = 1.0f;
            t = 0.5f / (float)Math.pow(weights.getColumns(), 1);
            return grow(weights);
        }

        return weights;
    }
    private FloatMatrix grow(FloatMatrix weights) {
        FloatMatrix newColumn = FloatMatrix.zeros(weights.getRows(), 1);

        return FloatMatrix.concatHorizontally(weights, newColumn);
    }
}
