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
    private FloatMatrix weights;
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
        final FloatMatrix meanVector = dataProvider.getMeanVectorForTraining();
        final FloatMatrix dataWithBias = dataProvider.getDataWithBiasForTraining();
        final FloatMatrix dataWithBiasTrans = dataProvider.getTransposedDataWithBiasForTraining();

//        show(weights.data, weights.rows, weights.columns, "weights");
//        show(dataWithBias.data, dataWithBias.rows, dataWithBias.columns, "data");
        // first associations
        FloatMatrix hidden = getHiddenFunction.get(dataWithBias, weights);
//        show(hidden.data, hidden.rows, hidden.columns, "hidden");

        final FloatMatrix positiveAssociations = new FloatMatrix(dataWithBiasTrans.rows, hidden.columns);
        ForkBlas.pmmuli(dataWithBiasTrans, hidden, positiveAssociations);
//        show(positiveAssociations.data, positiveAssociations.rows, positiveAssociations.columns, "positive");

        // guessed Data
        FloatMatrix visible = getVisibleFunction.get(hidden, weights.transpose());
        visible.subiColumnVector(meanVector);
        visible.putColumn(0, FloatMatrix.ones(visible.getRows(), 1));
//        show(visible.data, dataWithBias.rows, dataWithBias.columns, "data");

        // second associations
        hidden = getHiddenFunction.get(visible, weights);
//        show(hidden.data, hidden.rows, hidden.columns, "hidden 2");

        final FloatMatrix negativeAssociations = new FloatMatrix(dataWithBiasTrans.rows, hidden.columns);
        ForkBlas.pmmuli(visible.transpose(), hidden, negativeAssociations);
//        show(negativeAssociations.data, positiveAssociations.rows, positiveAssociations.columns, "negative");

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
        FloatMatrix visible = getVisibleFunction.get(hidden, weights.transpose()).subColumnVector(data.getMeanVector());

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


    public static void show(float[] data, int rows, int columns, String name)
    {
        System.out.println(name);
        for (int i = 0; i < columns; i++) {
            for (int j = 0; j < rows; j++) {
                System.out.print(String.format("%.5f", data[i * rows + j]) + ", ");
            }
            System.out.println();
        }
        System.out.println();
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
