package berlin.iconn.rbm.enhancements;

/**
 * Created by Moritz on 4/15/2014.
 */

import berlin.iconn.rbm.IRBM;
import berlin.iconn.rbm.StoppingCondition;
import berlin.iconn.rbm.dataprovider.ATrainingDataProvider;
import berlin.iconn.rbm.learningRate.ILearningRate;

import java.util.LinkedList;


public class RBMEnhancer implements IRBM {

    private final IRBM rbm;
    private final LinkedList<IRBMTrainingEnhancement> traningEnhancements;
    private final LinkedList<IRBMEndTrainingEnhancement> endEnhancements;
    private final RBMInfoPackage info;
    public final static int BASE_INTERVAL = 100;


    public RBMEnhancer(IRBM rbm) {
        super();
        this.rbm = rbm;
        this.traningEnhancements = new LinkedList<>();
        this.endEnhancements = new LinkedList<>();
        this.info = new RBMInfoPackage(0, rbm.getWeights(), 0);
    }

    public boolean addEnhancement(IRBMEnhancement enhancement) {

        boolean added = false;

        if (enhancement instanceof IRBMTrainingEnhancement) {

            traningEnhancements.add((IRBMTrainingEnhancement) enhancement);
            added = true;
        }

        if (enhancement instanceof IRBMEndTrainingEnhancement) {
            endEnhancements.add((IRBMEndTrainingEnhancement) enhancement);
            added = true;
        }

        return added;
    }

    @Override
    public void train(ATrainingDataProvider trainingData, StoppingCondition stop, ILearningRate learningRate) {

        setInfo(rbm, trainingData, 0);
        for (IRBMEnhancement enhancement : this.traningEnhancements) {
            enhancement.action(this.info);
        }

        boolean updateModel;
        while (stop.isNotDone()) {
            updateModel = true;

            StoppingCondition intervalStop = new StoppingCondition(
                    stop.isErrorDisabled(),
                    false,
                    stop.epochsRemaining() > BASE_INTERVAL
                            ? (stop.getCurrentEpochs() + BASE_INTERVAL) : stop.getMaxEpochs(),
                    stop.getMinError(),
                    stop.getCurrentEpochs(),
                    stop.getCurrentError());

            rbm.train(trainingData, intervalStop, learningRate);

            for (IRBMTrainingEnhancement enhancement : this.traningEnhancements) {
                if (intervalStop.getCurrentEpochs() % enhancement.getUpdateInterval() == 0) {
                    if (updateModel) {
                        updateModel = false;
                        setInfo(rbm, trainingData, intervalStop.getCurrentEpochs());
                    }
                    enhancement.action(this.info);
                }
            }
            stop.setCurrentEpochs(intervalStop.getCurrentEpochs());
            stop.setCurrentError(intervalStop.getCurrentError());
        }
        setInfo(rbm, trainingData, stop.getMaxEpochs());
        for (IRBMEndTrainingEnhancement enhancement : this.endEnhancements) {
            enhancement.action(this.info);
        }
    }

    private void setInfo(IRBM rbm, ATrainingDataProvider trainingData, int epochs) {
        this.info.setError(rbm.getError(trainingData.getData().toArray2()));
        this.info.setWeights(rbm.getWeights());
        this.info.setEpochs(epochs);
    }

    @Override
    public float getError(float[][] trainingData) {
        return rbm.getError(trainingData);
    }

    @Override
    public float[][] getHidden(float[][] userData) {
        return rbm.getHidden(userData);
    }

    @Override
    public float[][] getVisible(float[][] hiddenData) {
        return rbm.getVisible(hiddenData);
    }

    @Override
    public float[][] getWeights() {
        return rbm.getWeights();
    }

}

