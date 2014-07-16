package berlin.iconn.rbm.dataprovider;

import berlin.iconn.rbm.IRBM;
import org.jblas.FloatMatrix;

/**
 * Created by Moritz on 6/25/2014.
 */
public class CascadeDataProvider extends ATrainingDataProvider {
    private final IRBM rbm;
    private final ATrainingDataProvider provider;

    public CascadeDataProvider(IRBM rbm, ATrainingDataProvider provider) {
        super(provider.getData());
        this.rbm = rbm;
        this.provider = provider;
        changeDataAtTraining();
    }

    @Override
    public void changeDataAtTraining() {
        setDataWithBias(null);
        provider.changeDataAtTraining();
        setData(new FloatMatrix(rbm.getHidden(provider.getData().toArray2())));
    }
}
