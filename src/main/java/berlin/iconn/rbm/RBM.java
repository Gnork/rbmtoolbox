package berlin.iconn.rbm;


import berlin.iconn.rbm.dataprovider.ATrainingDataProvider;
import berlin.iconn.rbm.dataprovider.FullTrainingDataProvider;
import berlin.iconn.rbm.learningRate.ILearningRate;
import berlin.iconn.rbm.weightmodifier.DefaultModifier;
import berlin.iconn.rbm.weightmodifier.IWeightsModifier;
import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;


/**
 * Created by Moritz on 4/27/2014.
 */
public class RBM  implements IRBM {
    protected FloatMatrix weights;
    private final GetStatesFunction getHiddenFunction;
    private final GetStatesFunction getVisibleFunction;
    private final IWeightsModifier modifier;
    public RBM(GetStatesFunction getHiddenFunction, GetStatesFunction getVisibleFunction, FloatMatrix weights, IWeightsModifier modifier) {
        this.getHiddenFunction = getHiddenFunction;
        this.getVisibleFunction = getVisibleFunction;
        this.weights = weights;
        this.modifier = modifier;
    }

    public RBM(GetStatesFunction statesFunction, FloatMatrix weights, IWeightsModifier modifier) {
        this(statesFunction,statesFunction, weights, modifier);
    }

    public RBM(FloatMatrix weights, IWeightsModifier modifier) {
        this(new GetStatesFunction(), weights, modifier);
    }
    public RBM(FloatMatrix weights) {
        this(new GetStatesFunction(), weights, new DefaultModifier());
    }

    private void updateWeights(ATrainingDataProvider dataProvider, ILearningRate learningRate) {
        final FloatMatrix dataWithBias = dataProvider.getDataWithBias();
        final FloatMatrix dataWithBiasTrans = dataWithBias.transpose();

//        Main.print(weights.toArray2(), "weights");
//        Main.print(dataWithBias.toArray2(), "data");
        // first associations
        FloatMatrix hidden = getHiddenFunction.get(dataWithBias, weights);
//        Main.print(hidden.toArray2(), "hidden");

        final FloatMatrix positiveAssociations = new FloatMatrix(dataWithBiasTrans.rows, hidden.columns);
        ForkBlas.pmmuli(dataWithBiasTrans, hidden, positiveAssociations);
//        Main.print(positiveAssociations.toArray2(), "positive");

        // guessed Data
        FloatMatrix visible = getVisibleFunction.get(hidden, weights.transpose());
        visible.putColumn(0, FloatMatrix.ones(visible.getRows(), 1));
//        Main.print(visible.toArray2(), "visible");

        // second associations
        hidden = getHiddenFunction.get(visible, weights);
//        Main.print(hidden.toArray2(), "hidden 2");

        final FloatMatrix negativeAssociations = new FloatMatrix(dataWithBiasTrans.rows, hidden.columns);
        ForkBlas.pmmuli(visible.transpose(), hidden, negativeAssociations);
//        Main.print(negativeAssociations.toArray2(), "negative");

        // contrastive divergence on weights // update
        weights.addi((positiveAssociations.sub(negativeAssociations))
                .mul(learningRate.getRate() / (float) dataProvider.getData().getRows()));

    }

    public FloatMatrix getHidden(FloatMatrix data) {
        FullTrainingDataProvider provider = new FullTrainingDataProvider(data);
        return removeBiasFromData(getHiddenFunction.get(provider.getDataWithBias(), weights));
    }


    public FloatMatrix getVisible(FloatMatrix data) {
        FullTrainingDataProvider provider = new FullTrainingDataProvider(data);
        return removeBiasFromData(getHiddenFunction.get(provider.getDataWithBias(), weights.transpose()));
    }

    public float getError(ATrainingDataProvider data) {
        FloatMatrix dataWithBias = data.getDataWithBias();
        FloatMatrix hidden = getHiddenFunction.get(dataWithBias, weights);
        FloatMatrix visible = getVisibleFunction.get(hidden, weights.transpose());

        return (float) Math.sqrt(MatrixFunctions.pow(dataWithBias.sub(visible), 2.0f).sum() / data.getData().length / weights.getRows());
    }

    @Override
    public void train(ATrainingDataProvider dataProvider, StoppingCondition stop, ILearningRate learningRate) {
            float error = getError(dataProvider);
            while(stop.isNotDone()) {
//                System.out.println();
//                System.out.println("Normal RBM Epoch: " + stop.getCurrentEpochs());
//                System.out.println();
                learningRate.changeRate(error);
                updateWeights(dataProvider, learningRate);
                weights = modifier.modify(weights, error, stop.getCurrentEpochs());
                dataProvider.changeDataAtTraining();

                error = getError(dataProvider);
                stop.update(error);
            }

    }

    @Override
    public float getError(float[][] data) {
        return getError(new FullTrainingDataProvider(new FloatMatrix(data)));
    }

    @Override
    public float[][] getHidden(float[][] data) {
        return getHidden(new FloatMatrix(data)).toArray2();
    }

    @Override
    public float[][] getVisible(float[][] data) {
        return getVisible(new FloatMatrix(data)).toArray2();
    }

    @Override
    public float[][] getWeights() {
        return getWeightsMatrix().toArray2();
    }

    public FloatMatrix getWeightsMatrix() {
        return this.weights;
    }
    private FloatMatrix removeBiasFromData(FloatMatrix data) {
        return data.getRange(0, data.getRows(), 1, data.getColumns());
    }
}
