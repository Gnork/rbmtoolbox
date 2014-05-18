package berlin.iconn.rbm.weightmodifier;

import org.jblas.FloatMatrix;

/**
 * Created by Moritz on 4/29/2014.
 */
public interface IWeightsModifier {


    FloatMatrix modify(FloatMatrix weights, float error, int currentEpochs);
}
