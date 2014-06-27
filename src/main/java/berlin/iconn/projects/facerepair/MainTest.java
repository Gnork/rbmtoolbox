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
import berlin.iconn.rbm.RBM;
import berlin.iconn.rbm.WeightsFactory;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;
import org.jblas.FloatMatrix;

/**
 *
 * @author christoph
 */
public class MainTest {
    private static final int EDGE_LENGTH = 64;
    
    private static final String IMAGES_TRAINED = "D:\\image_sets\\rbm_face_images_png\\1000_images_trained";
    private static final String IMAGES_TRAINED_INCOMPLETE = "D:\\image_sets\\rbm_face_images_png\\1000_images_trained_incomplete_gray";
    private static final String IMAGES_NOT_TRAINED = "D:\\image_sets\\rbm_face_images_png\\1000_images_not_trained";
    private static final String IMAGES_NOT_TRAINED_INCOMPLETE = "D:\\image_sets\\rbm_face_images_png\\1000_images_not_trained_incomplete_gray";
    
    private static final String RBM1_WEIGHTS = "Output/SimpleWeights/WildFaces_64x64_rgb_5kh_3490it.dat";
    
    public static void main(String[] args){
        FloatMatrix rbm1Weights;
        try {
            rbm1Weights = new FloatMatrix(InOutOperations.loadSimpleWeights(new File(RBM1_WEIGHTS)));
        } catch (IOException | ClassNotFoundException ex) {
            Logger.getLogger(MainTest.class.getName()).log(Level.SEVERE, null, ex);
            return;
        }   
        RBM rbm1 = new RBM(rbm1Weights);
        RBM[] rbms = new RBM[]{rbm1};
        System.out.println("RBMs loaded");
        
        DataSet[] testTrainedSet = null;
        DataSet[] testNotTrainedSet = null;
        
        try {
            testTrainedSet = InOutOperations.loadImages(new File(IMAGES_TRAINED), 64, false, false, 0.0f, 1.0f, true);
            testNotTrainedSet = InOutOperations.loadImages(new File(IMAGES_NOT_TRAINED), 64, false, false, 0.0f, 1.0f, true);
        } catch (IOException ex) {
            Logger.getLogger(MainTest.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        float[][][] preparedTrained = prepareData(testTrainedSet, 10);
        float[][][] preparedNotTrained = prepareData(testNotTrainedSet, 10);
        
        
        try {
            //dynamicReconstructionTest(rbms, 64, preparedTrained, "IMAGES_TRAINED_DYNAMIC_1_reset", 1, true, false);
            //dynamicReconstructionTest(rbms, 64, preparedNotTrained, "IMAGES_NOT_TRAINED_DYNAMIC_1_reset", 1, true, false);
            //dynamicReconstructionTest(rbms, 64, preparedTrained, "IMAGES_TRAINED_DYNAMIC_10_reset_extra", 10, true, true);
            //dynamicReconstructionTest(rbms, 64, preparedNotTrained, "IMAGES_NOT_TRAINED_DYNAMIC_10_reset_extra", 10, true, true);
            //dynamicReconstructionTest(rbms, 64, preparedTrained, "IMAGES_TRAINED_DYNAMIC_10_reset", 10, true, false);
            //dynamicReconstructionTest(rbms, 64, preparedNotTrained, "IMAGES_NOT_TRAINED_DYNAMIC_10_reset", 10, true, false);
            //dynamicReconstructionTest(rbms, 64, preparedNotTrained, "IMAGES_NOT_TRAINED_DYNAMIC_100_reset", 100, true, false);
            dynamicReconstructionTest(rbms, 64, preparedTrained, "IMAGES_TRAINED_DYNAMIC_1_reset", 1, true, false);
        } catch (IOException ex) {
            Logger.getLogger(MainTest.class.getName()).log(Level.SEVERE, null, ex);
        }     
    }
    
    private static float[][][] prepareData(DataSet[] data, int numOfSamples){
        numOfSamples = Math.min(numOfSamples, data.length);
        float[][] compareData = new float[numOfSamples][];
        float[][] testData = new float[numOfSamples][];

        for(int i = 0; i < numOfSamples; ++i){
            testData[i] = data[i].getData();
            compareData[i] = new float[testData[i].length];
            System.arraycopy(testData[i], 0, compareData[i], 0, testData[i].length);
            float[] rgb = rgbMeanOfRegion(testData[i], 32, 64, 0, 64, 64);
            writeRGBtoRegion(testData[i], rgb[0], rgb[1], rgb[2], 0, 32, 0, 64, 64);
        }
        
        return new float[][][]{testData, compareData};
    }
    
    private static void dynamicReconstructionTest(RBM[] rbms, int edgeLength, float[][][] data, String testName, int numOfDreams, boolean resetOriginalPixels, boolean extraDream) throws IOException{
        System.out.println("Starting Test: " + testName);
        
        float[][] compareDataFloat = data[1];
        FloatMatrix reconData = new FloatMatrix(data[0]);
        for(int j = 0; j < numOfDreams; ++j){
            for(int i = 0; i < rbms.length; ++i){
                reconData = rbms[i].getHidden(reconData);
            }
            for(int i = rbms.length - 1; i >= 0; --i){
                reconData = rbms[i].getVisible(reconData);
            }
            
            if(resetOriginalPixels){
                float[][] reconDataFloat = reconData.toArray2();
                for(int i = 0; i < reconDataFloat.length; ++i){
                    replaceReconstructionWithOriginalPixels(reconDataFloat[i], compareDataFloat[i], 32, 64, 0, 64, 64);
                }
                reconData = new FloatMatrix(reconDataFloat);
            }
        }
        
        if(extraDream){
            for(int i = 0; i < rbms.length; ++i){
                reconData = rbms[i].getHidden(reconData);
            }
            for(int i = rbms.length - 1; i >= 0; --i){
                reconData = rbms[i].getVisible(reconData);
            }
        }
        
        float[][] reconDataFloat = reconData.toArray2();
        
        System.out.println(compareDataFloat[0].length);
        System.out.println(reconDataFloat[0].length);
        
        compareArraysForError(reconDataFloat, compareDataFloat, testName);
    }
    
    private static void replaceReconstructionWithOriginalPixels(float[] reconData, float[] originalData, int xStart, int xEnd, int yStart, int yEnd, int edgeLength){
        for(int i = yStart; i < yEnd; ++i){
            int yPos = i * 3;
            for(int j = xStart; j < xEnd; ++j){
                int xPos = j * 3;
                int pos = yPos * edgeLength + xPos;
                reconData[pos] = originalData[pos];
                reconData[pos + 1] = originalData[pos + 1];
                reconData[pos + 2] = originalData[pos + 2];
            }
        }
    }
    
    private static float[] rgbMeanOfRegion(float[] image, int xStart, int xEnd, int yStart, int yEnd, int edgeLength){
        float R = 0.0f;
        float G = 0.0f;
        float B = 0.0f;
        
        for(int i = yStart; i < yEnd; ++i){
            int yPos = i * 3;
            for(int j = xStart; j < xEnd; ++j){
                int xPos = j * 3;
                int pos = yPos * edgeLength + xPos;
                R += image[pos];
                G += image[pos + 1];
                B += image[pos + 2];
            }
        }
        int numOfPixels = (yEnd - yStart) * (xEnd - xStart);
        R /= numOfPixels;
        G /= numOfPixels;
        B /= numOfPixels;
        
        return new float[]{R, G, B};
    }
    
    private static float[] rgbMean(float[] image){
        int len = image.length / 3;
        float R = 0.0f;
        float G = 0.0f;
        float B = 0.0f;
        
        for(int i = 0; i < len; ++i){
            int pos = i * 3;
            R += image[pos];
            G += image[pos + 1];
            B += image[pos + 2];
        }
        R /= len;
        G /= len;
        B /= len;
        
        return new float[]{R, G, B};
    }
    
    private static void writeRGBtoRegion(float[] image, float R, float G, float B, int xStart, int xEnd, int yStart, int yEnd, int edgeLength){
        for(int i = yStart; i < yEnd; ++i){
            int yPos = i * 3;
            for(int j = xStart; j < xEnd; ++j){
                int xPos = j * 3;
                int pos = yPos * edgeLength + xPos;
                image[pos] = R;
                image[pos + 1] = G;
                image[pos + 2] = B;
            }
        }
    }
    
    private static float[][] copyFloatArray2(float[][] A){
        float[][] R = new float[A.length][];
        for(int i = 0; i < A.length; ++i){
            System.arraycopy(A[i], 0, R[i], 0, A.length);
        }
        return R;
    }
    
    private static void reconstructionTest(RBM[] rbms, int edgeLength, File testData, File compareData, String testName) throws IOException{
        System.out.println("Starting Test: " + testName);
        DataSet[] testDataSet = null;
        DataSet[] compareDataSet = null;
        try {
            testDataSet = InOutOperations.loadImages(testData, edgeLength, false, false, 0.0f, 1.0f, true);
            compareDataSet = InOutOperations.loadImages(compareData, edgeLength, false, false, 0.0f, 1.0f, true);
        } catch (IOException ex) {
            Logger.getLogger(MainTest.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        if(testDataSet.length != compareDataSet.length){
            System.out.println("test data length != compare data length");
            return;
        }
        
        float[][] testDataFloat = new float[testDataSet.length][];
        float[][] compareDataFloat = new float[compareDataSet.length][];
        for(int i = 0; i < testDataSet.length; ++i){
            testDataFloat[i] = testDataSet[i].getData();
            compareDataFloat[i] = compareDataSet[i].getData();
        }
        
        FloatMatrix reconData = new FloatMatrix(testDataFloat);
        for(int j = 0; j < 1; ++j){
            for(int i = 0; i < rbms.length; ++i){
                reconData = rbms[i].getHidden(reconData);
            }
            for(int i = rbms.length - 1; i >= 0; --i){
                reconData = rbms[i].getVisible(reconData);
            }
        }
        
        float[][] reconDataFloat = reconData.toArray2();
        
        compareArraysForError(reconDataFloat, compareDataFloat, testName);
    }
    
    private static void compareArraysForError(float[][] reconData, float[][] compareData, String testName) throws IOException{
        String dirString = "Output/Tests/" + testName;
        InOutOperations.mkdir(dirString);
        
        FileWriter writer = null;
        writer = new FileWriter(dirString + "/results.txt");

        String newLine = System.getProperty("line.separator");
        
        float finalMeanError = 0.0f;
        for(int i = 0; i < reconData.length; ++i){
            float imageError = 0.0f;
            for(int j = 0; j < reconData[i].length; ++j){
                imageError += Math.abs(reconData[i][j] - compareData[i][j]);
            }
            imageError /= reconData[i].length;
            String errorOut = "image " + (i+1) + " error: " + imageError;
            
            System.out.println(errorOut);
            writer.write(errorOut + newLine);

            finalMeanError += imageError;
            
            BufferedImage bi = DataConverter.pixelDataToImage(reconData[i], 0.0f, true);
            File imageOut = new File(dirString + "/recon" + i + ".png");
            ImageIO.write(bi, "png", imageOut);        
        }
        
        finalMeanError /= reconData.length;
        
        String finalOut = "final mean error: " + finalMeanError;
        System.out.println(finalOut);       
        writer.write(finalOut + newLine);
        writer.close();
    }
}
