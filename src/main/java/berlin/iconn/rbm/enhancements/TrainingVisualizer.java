package berlin.iconn.rbm.enhancements;

import berlin.iconn.rbm.enhancements.visualizations.IVisualizeObserver;

/**
 * Created by Moritz on 4/15/2014.
 */
public class TrainingVisualizer implements IRBMTrainingEnhancement {

    private final int updateInterval;
    private final IVisualizeObserver visObserver;

    public TrainingVisualizer(int updateInterval, IVisualizeObserver visObserver) {
        super();
        this.updateInterval = updateInterval;
        this.visObserver = visObserver;
    }

    @Override
    public void action(RBMInfoPackage info) {

        this.visObserver.update(info);

    }

    @Override
    public int getUpdateInterval() {
        return updateInterval;
    }
}
