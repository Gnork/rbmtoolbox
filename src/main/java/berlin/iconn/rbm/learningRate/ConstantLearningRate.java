package berlin.iconn.rbm.learningRate;

/**
 * Created by Moritz on 4/27/2014.
 */
public class ConstantLearningRate implements ILearningRate {

    private final float learningRate;

    public ConstantLearningRate(float learningRate) {
        this.learningRate = learningRate;
    }

    @Override
    public float getRate() {
        return learningRate;
    }

    @Override
    public void changeRate(float error) {

    }
}
