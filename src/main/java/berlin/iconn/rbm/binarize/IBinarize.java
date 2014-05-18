package berlin.iconn.rbm.binarize;

import org.jblas.FloatMatrix;

/**
 * Created by Moritz on 4/27/2014.
 */
@FunctionalInterface
public interface IBinarize {

    public FloatMatrix binarize(FloatMatrix data);
}
