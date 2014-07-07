package berlin.iconn.rbm.enhancements;

import berlin.iconn.rbm.IRBM;
import berlin.iconn.rbm.dataprovider.ATrainingDataProvider;
import org.jblas.FloatMatrix;

/**
 * Created by Moritz on 7/6/2014.
 */
public class CrossValidator implements IRBMTrainingEnhancement {
    private final IRBM rbm;
    private final ATrainingDataProvider crossValidationData;
    private final int validationCount;

    public CrossValidator(IRBM rbm, ATrainingDataProvider crossValidationData, int validationCount) {
        this.rbm = rbm;
        this.crossValidationData = crossValidationData;
        this.validationCount = validationCount;
    }

    @Override
    public int getUpdateInterval() {
        return 1;
    }

    @Override
    public void action(RBMInfoPackage info) {
        float sum = 0;
        for (int i = 0; i < validationCount; i++) {
            FloatMatrix data = crossValidationData.getData();
            FloatMatrix visible = new FloatMatrix(rbm.getVisible(rbm.getHidden(data.toArray2())));
            FloatMatrix diff = data.sub(visible);
            sum += (float) Math.sqrt(diff.mul(diff).sum() / data.length);
            crossValidationData.changeDataAtTraining();
        }
        System.out.println(sum / validationCount);
    }
}
