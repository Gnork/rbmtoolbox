package berlin.iconn.rbm;

import berlin.iconn.matrixExperiments.PlaygroundRBM;
import berlin.iconn.rbm.dataprovider.ATrainingDataProvider;
import berlin.iconn.rbm.dataprovider.FullTrainingDataProvider;
import berlin.iconn.rbm.learningRate.ConstantLearningRate;
import org.jblas.FloatMatrix;

public class Main {


    public static void main(String[] args) {
        float[][] data = {
                {0, 0, 0, 1, 1},
                {0, 0, 1, 1, 1},
                {1, 0, 0, 1, 1},
                {1, 1, 0, 0, 0},
                {1, 1, 1, 0, 0},
                {1, 1, 0, 0, 1}};

        FloatMatrix weights = new FloatMatrix(new float[][]{
                {0.0f, 0.0f, 0.0f},
                {0.0f, 0.003931488f, 0.0066873333f},
                {0.0f, 0.006410535f, -8.955293E-4f},
                {0.0f, -0.008205484f, 0.005057891f},
                {0.0f, 0.010423293f, 0.0072114877f},
                {0.0f, -0.014295372f, 0.006884049f}});

        IRBM rbm = new CudaRBM(weights);
        ATrainingDataProvider provider = new FullTrainingDataProvider(data);

        float[][] sampleData =  provider.getData().toArray2();


        long start = System.currentTimeMillis();
        rbm.train(provider, new StoppingCondition(1000), new ConstantLearningRate(0.1f));
        System.out.println(System.currentTimeMillis() - start);

        print(sampleData, "data");
        float[][] hidden = rbm.getHidden(sampleData);
        print(hidden, "hidden");
        float[][] visible = rbm.getVisible(hidden);
        print(visible, "visible");
    }


    public static void print(float[][] data, String name) {
        System.out.println(name);
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[0].length; j++) {
                System.out.print(String.format("%8.5f",data[i][j]) + ", ");
            }
            System.out.println();
        }
        System.out.println();
    }
}
