package berlin.iconn.rbm;

import org.jblas.FloatMatrix;

/**
 * Created by Moritz on 4/28/2014.
 */
@FunctionalInterface
public interface ILogistic {

    public FloatMatrix apply(FloatMatrix m);
}
