package berlin.iconn.matrixExperiments;

//import berlin.iconn.rbm.Main;
import berlin.iconn.rbm.RBM;
import berlin.iconn.rbm.StoppingCondition;
import berlin.iconn.rbm.dataprovider.ATrainingDataProvider;
import berlin.iconn.rbm.learningRate.ILearningRate;
import org.jblas.FloatMatrix;

/**
 * Created by Moritz on 6/14/2014.
 */
public class PlaygroundRBM extends RBM {


    Matrix positive;
    Matrix negative;
    float transition = 0.1f;
    public PlaygroundRBM(FloatMatrix weights) {
        super(weights);
        positive = new Matrix(getWeights());
        negative = new Matrix(getWeights());
    }

    @Override
    public void train(ATrainingDataProvider dataProvider, StoppingCondition stop, ILearningRate learningRate) {

        Matrix weights = new Matrix(getWeights());

        while (stop.isNotDone()) {
            Matrix dataWithBias = new Matrix(dataProvider.getDataWithBiasForTraining().toArray2());
            Matrix dataWithBiasT = dataWithBias.transpose();
//            if(transition <= 1.0f) {
//                dataWithBias = dataWithBias.subtract(dataWithBias.multiplyWithRandom().multiply(1 - transition));
//                dataWithBiasT = dataWithBiasT.subtract(dataWithBiasT.multiplyWithRandom().multiply(1 - transition));
//                transition += learningRate.getRate() * 0.001f;
//            }

//            Main.print(weights.toArray2(), "weights");
//            Main.print(dataWithBias.toArray2(), "data");
            Matrix hidden = dataWithBias.matrixMultiply(weights);
            hidden.iApplyLogistic();
//            Main.print(hidden.toArray2(), "hidden");

            positive = dataWithBiasT.matrixMultiply(hidden);
//            Main.print(positive.toArray2(), "positive");

            Matrix visible = hidden.matrixMultiply(weights.transpose());
            visible.iApplyLogistic();
            visible.resetDataBias();
//            Main.print(visible.toArray2(), "visible");

            hidden = visible.matrixMultiply(weights);
            hidden.iApplyLogistic();
//            Main.print(hidden.toArray2(), "hidden 2");

            negative = visible.transpose().matrixMultiply(hidden);
//            Main.print(negative.toArray2(), "negative");

            weights = weights.plus(positive.subtract(negative).scale(learningRate.getRate() / dataWithBias.getRows()));
            stop.update(dataWithBias.getMeanSquareError(visible));
            dataProvider.changeDataAtTraining();

        }
        this.weights = new FloatMatrix(weights.toArray2());
    }
}
