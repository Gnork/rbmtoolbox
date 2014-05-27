/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package berlin.iconn.projects.segmentation;

import berlin.iconn.persistence.InOutOperations;
import berlin.iconn.rbm.*;
import berlin.iconn.rbm.learningRate.ILearningRate;
import java.io.IOException;
import java.util.Date;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.jblas.FloatMatrix;

/**
 *
 * @author christoph
 */
public class RBMSegmentationStack{
    private final int weightsSaverInterval = Integer.MAX_VALUE;
    private final int baseInterval = 1000;
    private final Date date = new Date();
    
    private IRBM labelRBM;
    private IRBM imageRBM;
    private IRBM combiRBM;
    
    public RBMSegmentationStack(int labelIn, int labelOut, int imageIn, int imageOut, int combiOut, float learningRate){
        this(   WeightsFactory.randomGaussianWeightsWithBias(labelIn, labelOut, learningRate),
                WeightsFactory.randomGaussianWeightsWithBias(imageIn, imageOut, learningRate),
                WeightsFactory.randomGaussianWeightsWithBias(labelOut + imageOut, combiOut, learningRate));
    }
    
    public RBMSegmentationStack(FloatMatrix labelWeights, FloatMatrix imageWeights, FloatMatrix combiWeights){
        labelRBM = new RBM(labelWeights);
        imageRBM = new RBM(imageWeights);
        combiRBM = new RBM(combiWeights);
        
        //Date date = new Date();
        //labelRBM.addEnhancement(new WeightsSaver(date, weightsSaverInterval, "label"));
        //imageRBM.addEnhancement(new WeightsSaver(date, weightsSaverInterval, "image"));
        //combiRBM.addEnhancement(new WeightsSaver(date, weightsSaverInterval, "combi"));
    }


    public void train(SegmentationStackRandomBatchGenerator dataProvider, StoppingCondition stop, ILearningRate learningRate) {
        
        SegmentationStackComponentProvider imageProvider = new SegmentationStackComponentProvider(dataProvider.getImageData());
        SegmentationStackComponentProvider labelProvider = new SegmentationStackComponentProvider(dataProvider.getLabelData());
        SegmentationStackComponentProvider combiProvider = new SegmentationStackComponentProvider(new FloatMatrix(dataProvider.getImageData().rows,1));
        
        stop = new StoppingCondition(stop.getMaxEpochs() / baseInterval);
        int allEpochs = stop.getMaxEpochs() * baseInterval;
        
        while(stop.isNotDone()){
            stop.update(0.0f);
            
            dataProvider.changeDataAtTraining();           
            FloatMatrix images = dataProvider.getImageData();
            FloatMatrix labels = dataProvider.getLabelData();
            imageProvider.setDataForTraining(images);
            labelProvider.setDataForTraining(labels);
            
            imageRBM.train(imageProvider, new StoppingCondition(baseInterval), learningRate);
            labelRBM.train(labelProvider, new StoppingCondition(baseInterval), learningRate);
            
            float[][] imageHidden = imageRBM.getHidden(images.toArray2());
            float[][] labelHidden = labelRBM.getHidden(labels.toArray2());            
            FloatMatrix combi = new FloatMatrix(concat(labelHidden, imageHidden));
            combiProvider.setDataForTraining(combi);
            
            combiRBM.train(combiProvider, new StoppingCondition(baseInterval), learningRate);
            
            
            // logging functions
            float imageError = imageRBM.getError(images.toArray2());
            float labelError = labelRBM.getError(labels.toArray2());
            float combiError = combiRBM.getError(combi.toArray2());
            int epochs = stop.getCurrentEpochs() * baseInterval;
            
            try {
                InOutOperations.saveSimpleWeights(imageRBM.getWeights(), date, "image");
                InOutOperations.saveSimpleWeights(labelRBM.getWeights(), date, "label");
                InOutOperations.saveSimpleWeights(combiRBM.getWeights(), date, "combi");
            } catch (IOException ex) {
                System.err.println("Could not save weights");
                Logger.getLogger(RBMSegmentationStack.class.getName()).log(Level.SEVERE, null, ex);
            }
            
            System.out.println("labels: " + labelError + "\timages: " + imageError + "\tcombi: " + combiError + "\tepochs: " + epochs + " / " + allEpochs);
        }
    }
    
    public float[] reconstructLabel(float[] imagePatch, int numOfClasses){
        int labelOut = labelRBM.getWeights()[0].length  - 1;
        float[][] zeroLabels = new float[1][numOfClasses];
        float[][] labelHidden = labelRBM.getHidden(zeroLabels);
        //float[][] imageHidden = imageRBM.getHidden(new float[][]{imagePatch});
        float[][] imageHidden = new float[1][labelOut];
        float[][] combiVisible = concat(labelHidden, imageHidden);
        float[][] combiHidden = combiRBM.getHidden(combiVisible);
        float[][] combiReconstruct = combiRBM.getVisible(combiHidden);       
        float[][] labelPart = new float[1][labelOut];
        System.arraycopy(combiReconstruct[0], 0, labelPart[0], 0, labelOut);
        float[][] labelReconstruct = labelRBM.getVisible(labelPart);
        return labelReconstruct[0];
    }
    
    private float[][] concat(float[][] A, float[][] B){
        float[][] R = new float[A.length][];
        for(int i = 0; i < R.length; ++i){
            R[i] = concat(A[i], B[i]);
        }
        return R;
    }
    
    private float[] concat(float[] A, float[] B) {
        float[] R = new float[A.length+B.length];
        System.arraycopy(A, 0, R, 0, A.length);
        System.arraycopy(B, 0, R, A.length, B.length);
        return R;
    }
}
