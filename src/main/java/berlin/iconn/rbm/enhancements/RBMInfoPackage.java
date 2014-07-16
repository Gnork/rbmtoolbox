package berlin.iconn.rbm.enhancements;

/**
 * Created by Moritz on 4/15/2014.
 */
import java.util.LinkedList;

/**
 *
 * @author moritz
 */
public class RBMInfoPackage {

    private float error;
    private float[][] weights;
    private int epochs;

    public RBMInfoPackage(float error, float[][] weights, int epochs) {
        this.error = error;
        this.weights = weights;
        this.epochs = epochs;
    }

    public float getError() {
        return error;
    }

    public float[][] getWeights() {
        return weights;
    }

    public int getEpochs() {
        return epochs;
    }


    /**
     * @param error the error to set
     */
    public void setError(float error) {
        this.error = error;
    }

    /**
     * @param weights the weights to set
     */
    public void setWeights(float[][] weights) {
        this.weights = weights;
    }

    /**
     * @param epochs the epochs to set
     */
    public void setEpochs(int epochs) {
        this.epochs = epochs;
    }
}
