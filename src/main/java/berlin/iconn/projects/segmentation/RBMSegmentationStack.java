/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package berlin.iconn.projects.segmentation;

import berlin.iconn.persistence.InOutOperations;
import berlin.iconn.rbm.*;
import berlin.iconn.rbm.binarize.CommonBinarize;
import berlin.iconn.rbm.learningRate.ILearningRate;
import berlin.iconn.rbm.weightmodifier.DefaultModifier;
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
    private IRBM assocRBM;
    
    public RBMSegmentationStack(int labelIn, int labelOut, FloatMatrix labelWeights, int imageIn, int imageOut, FloatMatrix imageWeights, int combiOut, FloatMatrix combiWeights, int assocOut, FloatMatrix assocWeights, float weightsFactor, boolean binarizeHidden){
        
        this(   labelWeights == null ? WeightsFactory.randomGaussianWeightsWithBias(labelIn, labelOut, weightsFactor) : labelWeights,
                imageWeights == null ? WeightsFactory.randomGaussianWeightsWithBias(imageIn, imageOut, weightsFactor): imageWeights,
                combiWeights == null ? WeightsFactory.randomGaussianWeightsWithBias(labelOut + imageOut, combiOut, weightsFactor): combiWeights,
                assocWeights == null ? WeightsFactory.randomGaussianWeightsWithBias(combiOut, assocOut, weightsFactor): assocWeights,
                binarizeHidden);
    }
    
    public RBMSegmentationStack(int labelIn, int labelOut, int imageIn, int imageOut, int combiOut, int assocOut, float weightsFactor, boolean binarizeHidden){
        this(   WeightsFactory.randomGaussianWeightsWithBias(labelIn, labelOut, weightsFactor),
                WeightsFactory.randomGaussianWeightsWithBias(imageIn, imageOut, weightsFactor),
                WeightsFactory.randomGaussianWeightsWithBias(labelOut + imageOut, combiOut, weightsFactor),
                WeightsFactory.randomGaussianWeightsWithBias(combiOut, assocOut, weightsFactor),
                binarizeHidden);
    }
    
    public RBMSegmentationStack(FloatMatrix labelWeights, FloatMatrix imageWeights, FloatMatrix combiWeights, FloatMatrix assocWeights, boolean binarizeHidden){
        if(binarizeHidden){
            labelRBM = new RBM(new GetStatesFunction(new CommonBinarize()), new GetStatesFunction(), labelWeights, new DefaultModifier());
            imageRBM = new RBM(new GetStatesFunction(new CommonBinarize()), new GetStatesFunction(), imageWeights, new DefaultModifier());
            combiRBM = new RBM(new GetStatesFunction(new CommonBinarize()), new GetStatesFunction(), combiWeights, new DefaultModifier());
            assocRBM = new RBM(new GetStatesFunction(new CommonBinarize()), new GetStatesFunction(), assocWeights, new DefaultModifier());
        }else{
            labelRBM = new RBM(labelWeights);
            imageRBM = new RBM(imageWeights);
            combiRBM = new RBM(combiWeights);
            assocRBM = new RBM(assocWeights);
        }     
    }
    
    public void trainLabels(FloatMatrix labelWeights, SegmentationStackRandomBatchGenerator generator, StoppingCondition stop, ILearningRate learningRate, boolean binarizeHidden){
        FloatMatrix weights = new FloatMatrix(labelRBM.getWeights());
        if(labelWeights != null){
            weights = labelWeights;
        }
        if(binarizeHidden){
            labelRBM = new RBM(new GetStatesFunction(new CommonBinarize()), new GetStatesFunction(), weights, new DefaultModifier());
        }else{
            labelRBM = new RBM(new GetStatesFunction(), new GetStatesFunction(), weights, new DefaultModifier());
        }
        stop = new StoppingCondition(stop.getMaxEpochs() / baseInterval);
        int allEpochs = stop.getMaxEpochs() * baseInterval;
        
        SegmentationStackComponentProvider labelProvider = new SegmentationStackComponentProvider(generator.getLabelData());
        
        while(stop.isNotDone()){
            stop.update(0.0f);
            generator.changeDataAtTraining();     
            labelProvider.setDataForTraining(generator.getLabelData());          
            labelRBM.train(labelProvider, new StoppingCondition(baseInterval), learningRate);
            
            float labelError = labelRBM.getError(generator.getLabelData().toArray2());
            int epochs = stop.getCurrentEpochs() * baseInterval;
            
            try {
                InOutOperations.saveSimpleWeights(labelRBM.getWeights(), date, "label");
            } catch (IOException ex) {
                System.err.println("Could not save weights");
                Logger.getLogger(RBMSegmentationStack.class.getName()).log(Level.SEVERE, null, ex);
            }          
            System.out.println("labels: " + labelError + "\tepochs: " + epochs + " / " + allEpochs);
        }      
    }


    public void train(SegmentationStackRandomBatchGenerator generator, StoppingCondition stop, ILearningRate learningRate) {
        
        SegmentationStackComponentProvider imageProvider = new SegmentationStackComponentProvider(generator.getImageData());
        SegmentationStackComponentProvider labelProvider = new SegmentationStackComponentProvider(generator.getLabelData());
        SegmentationStackComponentProvider combiProvider = new SegmentationStackComponentProvider(new FloatMatrix(generator.getImageData().rows,1));
        SegmentationStackComponentProvider assocProvider = new SegmentationStackComponentProvider(new FloatMatrix(generator.getImageData().rows,1));
        
        stop = new StoppingCondition(stop.getMaxEpochs() / baseInterval);
        int allEpochs = stop.getMaxEpochs() * baseInterval;
        
        while(stop.isNotDone()){
            stop.update(0.0f);
            
            generator.changeDataAtTraining();           
            FloatMatrix images = generator.getImageData();
            FloatMatrix labels = generator.getLabelData();
            imageProvider.setDataForTraining(images);
            labelProvider.setDataForTraining(labels);          
            imageRBM.train(imageProvider, new StoppingCondition(baseInterval), learningRate);
            labelRBM.train(labelProvider, new StoppingCondition(baseInterval), learningRate);
            
            float[][] imageHidden = imageRBM.getHidden(images.toArray2());
            float[][] labelHidden = labelRBM.getHidden(labels.toArray2());
            float[][] concatHidden = concat(labelHidden, imageHidden);
            combiProvider.setDataForTraining(new FloatMatrix(concatHidden));          
            combiRBM.train(combiProvider, new StoppingCondition(baseInterval), learningRate);
            
            float[][] combiHidden = combiRBM.getHidden(concatHidden);
            assocProvider.setDataForTraining(new FloatMatrix(combiHidden));
            assocRBM.train(assocProvider, new StoppingCondition(baseInterval), learningRate);
            
            
            // logging functions
            float imageError = imageRBM.getError(images.toArray2());
            float labelError = labelRBM.getError(labels.toArray2());
            float combiError = combiRBM.getError(concatHidden);
            float assocError = assocRBM.getError(combiHidden);
            int epochs = stop.getCurrentEpochs() * baseInterval;
            
            try {
                InOutOperations.saveSimpleWeights(imageRBM.getWeights(), date, "image");
                InOutOperations.saveSimpleWeights(labelRBM.getWeights(), date, "label");
                InOutOperations.saveSimpleWeights(combiRBM.getWeights(), date, "combi");
                InOutOperations.saveSimpleWeights(assocRBM.getWeights(), date, "assoc");
            } catch (IOException ex) {
                System.err.println("Could not save weights");
                Logger.getLogger(RBMSegmentationStack.class.getName()).log(Level.SEVERE, null, ex);
            }
            
            System.out.println("labels: " + labelError + "\timages: " + imageError + "\tcombi: " + combiError + "\tassoc: " + assocError + "\tepochs: " + epochs + " / " + allEpochs);
        }
    }
    
    public float[] reconstructLabel(float[] imagePatch, int numOfClasses){
        int labelOut = labelRBM.getWeights()[0].length  - 1;
        float[][] zeroLabels = new float[1][numOfClasses];
        float[][] labelHidden = labelRBM.getHidden(zeroLabels);
        float[][] imageHidden = imageRBM.getHidden(new float[][]{imagePatch});
        //float[][] imageHidden = new float[1][labelOut];
        float[][] concatHidden = concat(labelHidden, imageHidden);
        float[][] combiHidden = combiRBM.getHidden(concatHidden);
        System.out.println("combiHidden " + combiHidden[0].length);
        System.out.println("assocRBM " + assocRBM.getWeights().length);
        float[][] assocHidden = assocRBM.getHidden(combiHidden);
        float[][] assocReconstruct = assocRBM.getVisible(assocHidden);
        float[][] combiReconstruct = combiRBM.getVisible(assocReconstruct);       
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
