package berlin.iconn.rbm;

import berlin.iconn.rbm.dataprovider.ATrainingDataProvider;
import berlin.iconn.rbm.dataprovider.FullTrainingDataProvider;
import berlin.iconn.rbm.learningRate.ConstantLearningRate;
import org.jblas.FloatMatrix;

import java.util.Random;

/**
 * Created by Moritz on 6/25/2014.
 */
public class Main2 {


    public static void main(String[] args) {
        Random random = new Random();
        float[][] data = new float[11][8];
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[0].length; j++) {
                data[i][j] = random.nextInt(2);
            }
        }

        FloatMatrix weights = WeightsFactory.patternWeightsWithBias(data[0].length, 100, 0.01f);

        IRBM rbm = new CudaRBM(weights);
        ATrainingDataProvider provider = new FullTrainingDataProvider(data);

        float[][] sampleData =  provider.getData().toArray2();


        long start = System.currentTimeMillis();
        rbm.train(provider, new StoppingCondition(0), new ConstantLearningRate(0.1f));
        System.out.println(System.currentTimeMillis() - start);

        print(sampleData, "data");
        float[][] hidden = rbm.getHidden(sampleData);
      //  print(hidden, "hidden");
        float[][] visible = rbm.getVisible(hidden);
        print(visible, "visible");
    }


    public static void print(float[][] data, String name) {
        System.out.println(name);
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[0].length; j++) {
                System.out.print(String.format("%8.2f",data[i][j]) + ", ");
            }
            System.out.println();
        }
        System.out.println();
    }
}
