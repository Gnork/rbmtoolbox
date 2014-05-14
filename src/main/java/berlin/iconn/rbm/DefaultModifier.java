package berlin.iconn.rbm;

import org.jblas.FloatMatrix;

/**
 * Created by Moritz on 4/29/2014.
 */
public class DefaultModifier implements ITrainingModifier {
    @Override
    public FloatMatrix modify(FloatMatrix weights, float error, int currentEpochs) {
        return  weights;
    }
}
