package berlin.iconn.rbm.learningRate;

/**
 * Created by Moritz on 4/27/2014.
 */
public interface ILearningRate {
    public float getRate();

    public void changeRate(float error);
}
