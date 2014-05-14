package berlin.iconn.rbm;

import berlin.iconn.rbm.dataprovider.ATrainingDataProvider;
import berlin.iconn.rbm.dataprovider.RandomPictureBatchSelectionProvider;
import berlin.iconn.rbm.learningRate.ConstantLearningRate;
import org.jblas.FloatMatrix;

public class Main {


    public static void main(String[] args) {
        float[][] data = {
                {1, 0, 0, 0, 0, 0, 1},
                {0, 1, 0, 0, 0, 1, 0},
                {0, 0, 1, 0, 1, 0, 0},
                {0, 0, 0, 1, 0, 0, 0},
                {0, 0, 1, 1, 1, 0, 0},
                {0, 1, 0, 1, 0, 1, 0},
                {1, 0, 1, 0, 1, 0, 1}};
        RBM rbm = new RBM(//new GetStatesFunction((mdata) -> (mdata.ge(FloatMatrix.rand(mdata.rows, mdata.columns)))), new GetStatesFunction(),
                WeightsFactory.randomGaussianWeightsWithBias(9, 18, 0.01f, 45));
        ATrainingDataProvider provider = new RandomPictureBatchSelectionProvider(new FloatMatrix[]{new FloatMatrix(data)},10, 3,3, 0);

        float[][] sampleData =  provider.getData().addiColumnVector(provider.getMeanVector()).toArray2();
        rbm.train(provider, new StoppingCondition(1_000_000), new ConstantLearningRate(0.001f));

        print(sampleData);
        float[][] hidden = rbm.getHidden(sampleData);
        float[][] visible = rbm.getVisible(hidden);
        print(visible);
    }


    public static void print(float[][] data) {
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[0].length; j++) {
                System.out.print( Math.round(data[i][j]) + ", ");
            }
            System.out.println();
        }
        System.out.println();
    }
}
