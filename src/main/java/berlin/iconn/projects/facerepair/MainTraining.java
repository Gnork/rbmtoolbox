/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package berlin.iconn.projects.facerepair;

import berlin.iconn.persistence.InOutOperations;
import berlin.iconn.rbm.CudaRBM;
import berlin.iconn.rbm.DataConverter;
import berlin.iconn.rbm.DataSet;
import berlin.iconn.rbm.NativeRBM;
import berlin.iconn.rbm.RBM;
import berlin.iconn.rbm.StoppingCondition;
import berlin.iconn.rbm.WeightsFactory;
import berlin.iconn.rbm.dataprovider.BatchTrainingDataProvider;
import berlin.iconn.rbm.enhancements.InfoLogger;
import berlin.iconn.rbm.enhancements.RBMEnhancer;
import berlin.iconn.rbm.enhancements.WeightsSaver;
import berlin.iconn.rbm.learningRate.ConstantLearningRate;
import java.io.File;
import java.io.IOException;
import java.util.Date;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.jblas.FloatMatrix;

/**
 *
 * @author christoph
 */
public class MainTraining {

    private static final int EDGE_LENGTH = 64;
    private static final String IMAGES = "D:\\image_sets\\rbm_face_images_png\\training_set";
    private static final String RBM1_WEIGHTS = "Output\\SimpleWeights\\2014_06_26_09_45_38_faces_cuda.dat";
    
    public static void main(String[] args) {

        DataSet[] trainingDataSet;
        FloatMatrix rbm1Weights;
        try {
            trainingDataSet = InOutOperations.loadImages(new File(IMAGES), 64, false, false, 0.0f, 1.0f, true);
            rbm1Weights = new FloatMatrix(InOutOperations.loadSimpleWeights(new File(RBM1_WEIGHTS)));
        } catch (IOException | ClassNotFoundException ex) {
            Logger.getLogger(MainTraining.class.getName()).log(Level.SEVERE, null, ex);
            return;
        }
        
        System.out.println("training set loaded");
        final float[][] trainingData = DataConverter.dataSetToArray(trainingDataSet);

        int inputSize = EDGE_LENGTH * EDGE_LENGTH * 3;
        //rbm1Weights = WeightsFactory.randomGaussianWeightsWithBias(inputSize, 6000, 0.01f);
        RBMEnhancer enhancer = new RBMEnhancer(new NativeRBM(rbm1Weights));
        
        WeightsSaver ws = new WeightsSaver(new Date(), 100, "faces_cuda");
        enhancer.addEnhancement(ws);
        InfoLogger il = new InfoLogger(100);
        enhancer.addEnhancement(il);

        BatchTrainingDataProvider provider = new BatchTrainingDataProvider(new FloatMatrix(trainingData), 100);
        enhancer.train( provider,
                        new StoppingCondition(1_000_000),
                        new ConstantLearningRate(0.01f));
    }
}
