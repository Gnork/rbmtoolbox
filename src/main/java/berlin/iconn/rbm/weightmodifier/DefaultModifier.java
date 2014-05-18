package berlin.iconn.rbm.weightmodifier;

import org.jblas.FloatMatrix;

/**
 * Created by Moritz on 4/29/2014.
 */
public class DefaultModifier implements IWeightsModifier {
    @Override
    public FloatMatrix modify(FloatMatrix weights, float error, int currentEpochs) {
        return  weights;
    }
}
