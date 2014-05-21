package berlin.iconn.rbm;


import berlin.iconn.rbm.dataprovider.ATrainingDataProvider;
import berlin.iconn.rbm.learningRate.ILearningRate;
import berlin.iconn.rbm.logistic.ILogistic;

import java.awt.image.BufferedImage;

public interface IRBM {

    public void train(ATrainingDataProvider dataProvider, StoppingCondition stop, ILearningRate learningRate);
    public float getError(float[][] data);
    public float[][] getHidden(float[][] data);
    public float[][] getVisible(float[][] data);

    public float[][] getWeights();



}
