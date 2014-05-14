package berlin.iconn.rbm;

import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;

/**
 * Created by Moritz on 4/27/2014.
 */
public class GetStatesFunction {
    private final static ILogistic common = (mData) -> {
        final FloatMatrix negM = mData.neg();
        final FloatMatrix negExpM = MatrixFunctions.exp(negM);
        final FloatMatrix negExpPlus1M = negExpM.add(1.0f);
        final FloatMatrix OneDivideNegExpPlusOneM = MatrixFunctions.pow(negExpPlus1M, -1.0f);
        return OneDivideNegExpPlusOneM;
    };
    private final ILogistic logistic;
    private final IBinarize binarize;

    public GetStatesFunction(ILogistic logistic, IBinarize binarize) {
        this.logistic = logistic;
        this.binarize = binarize;
    }


    public GetStatesFunction() {
        this.logistic = common;
        this.binarize = (mData) -> mData;
    }

    public GetStatesFunction(IBinarize binarize) {
        this.logistic = common;
        this.binarize = binarize;
    }

    public FloatMatrix get(FloatMatrix data, FloatMatrix weights) {
        FloatMatrix mData = new FloatMatrix(data.getRows(), weights.getColumns());
        ForkBlas.pmmuli(data,weights,mData);
        return binarize.binarize(logistic.apply(mData));
    }
}
