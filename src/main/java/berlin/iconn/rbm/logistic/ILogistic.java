package berlin.iconn.rbm.logistic;

import org.jblas.FloatMatrix;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;

/**
 * Created by Moritz on 4/28/2014.
 */
@FunctionalInterface
public interface ILogistic {

    public FloatMatrix apply(FloatMatrix m);

}
