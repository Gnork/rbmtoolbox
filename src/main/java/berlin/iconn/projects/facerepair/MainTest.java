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
    
    private static final String RBM1000_WEIGHTS = "Output/SimpleWeights/output_2300it_TE13,544_CVE14,172_gray.dat";
    
    public static void main(String[] args){
        FloatMatrix rbm1000Weights;
        try {
            rbm1000Weights = new FloatMatrix(InOutOperations.loadSimpleWeights(new File(RBM1000_WEIGHTS)));
        } catch (IOException | ClassNotFoundException ex) {
            Logger.getLogger(MainTest.class.getName()).log(Level.SEVERE, null, ex);
            return;
        }   
        RBM rbm1000 = new RBM(rbm1000Weights);
        RBM[] rbms = new RBM[]{rbm1000};
        System.out.println("RBMs loaded");
        
        DataSet[] testTrainedSet = null;
        DataSet[] testNotTrainedSet = null;
        
        int numOfSamples = 30;
        int numOfEpochs = 3;
        
        boolean resetOriginalPixels = true;
        boolean extraEpoch = false;
        boolean rgb = false;
        
        try {
            testTrainedSet = InOutOperations.loadImages(new File(IMAGES_TRAINED), 64, false, false, 0.0f, 1.0f, rgb);
            testNotTrainedSet = InOutOperations.loadImages(new File(IMAGES_NOT_TRAINED), 64, false, false, 0.0f, 1.0f, rgb);
        } catch (IOException ex) {
            Logger.getLogger(MainTest.class.getName()).log(Level.SEVERE, null, ex);
        }  
        
        float[][][] imagesTrained = prepareData(testTrainedSet, numOfSamples, rgb);
        float[][][] imagesNotTrained = prepareData(testNotTrainedSet, numOfSamples, rgb);
           
        try {
            //dynamicReconstructionTest(rbms, 64, imagesNotTrained, "IMAGES_NOT_TRAINED_" + 1 + "e_gray", 1, resetOriginalPixels, extraEpoch, rgb);
            //dynamicReconstructionTest(rbms, 64, imagesNotTrained, "IMAGES_NOT_TRAINED_" + 3 + "e_gray", 3, resetOriginalPixels, extraEpoch, rgb);
            //dynamicReconstructionTest(rbms, 64, imagesNotTrained, "IMAGES_NOT_TRAINED_" + 5 + "e_gray", 5, resetOriginalPixels, extraEpoch, rgb);
            dynamicReconstructionTest(rbms, 64, imagesTrained, "IMAGES_TRAINED_" + 1 + "e_gray", 1, resetOriginalPixels, extraEpoch, rgb);
            //dynamicReconstructionTest(rbms, 64, imagesTrained, "IMAGES_TRAINED_" + numOfEpochs + "e_reset_extra", numOfEpochs, true, true);
            //dynamicReconstructionTest(rbms, 64, imagesNotTrained, "IMAGES_NOT_TRAINED_" + numOfEpochs + "e_reset", numOfEpochs, true, false);
            //dynamicReconstructionTest(rbms, 64, imagesNotTrained, "IMAGES_NOT_TRAINED_" + numOfEpochs + "e_reset_extra", numOfEpochs, true, true);
        } catch (IOException ex) {
            Logger.getLogger(MainTest.class.getName()).log(Level.SEVERE, null, ex);
        }     
    }
    
    private static float[][][] prepareData(DataSet[] data, int numOfSamples, boolean rgb){
        numOfSamples = Math.min(numOfSamples, data.length);
        float[][] compareData = new float[numOfSamples][];
        float[][] testData = new float[numOfSamples][];

        for(int i = 0; i < numOfSamples; ++i){
            testData[i] = data[i].getData();
            compareData[i] = new float[testData[i].length];
            System.arraycopy(testData[i], 0, compareData[i], 0, testData[i].length);
            if(rgb){
                float[] meanRGB = rgbMeanOfRegion(testData[i], 32, 64, 0, 64, 64);
                writeRGBtoRegion(testData[i], meanRGB[0], meanRGB[1], meanRGB[2], 0, 32, 0, 64, 64);
            }else{
                float meanL = luminaceMeanOfRegion(testData[i], 32, 64, 0, 64, 64);
                writeLtoRegion(testData[i], meanL, 0, 32, 0, 64, 64);
            }
        }
        
        return new float[][][]{testData, compareData};
    }
    
    private static void specialReconstructionTest(RBM rbm1000, RBM rbm2000, int edgeLength, float[][][] data, String testName, int epochs1000, int epochs2000) throws IOException{
        System.out.println("Starting Test: " + testName);
        
        float[][] compareDataFloat = data[1];
        FloatMatrix reconData = new FloatMatrix(data[0]);
        for(int j = 0; j < epochs1000; ++j){
            
            reconData = rbm1000.getHidden(reconData);
            reconData = rbm1000.getVisible(reconData);
            
            float[][] reconDataFloat = reconData.toArray2();
            for(int i = 0; i < reconDataFloat.length; ++i){
                replaceReconstructionWithOriginalPixels(reconDataFloat[i], compareDataFloat[i], 32, 64, 0, 64, 64, true);
            }
            reconData = new FloatMatrix(reconDataFloat);
        }
        
        for(int j = 0; j < epochs2000; ++j){
            
            reconData = rbm2000.getHidden(reconData);
            reconData = rbm2000.getVisible(reconData);
            
            float[][] reconDataFloat = reconData.toArray2();
            for(int i = 0; i < reconDataFloat.length; ++i){
                replaceReconstructionWithOriginalPixels(reconDataFloat[i], compareDataFloat[i], 32, 64, 0, 64, 64, true);
            }
            reconData = new FloatMatrix(reconDataFloat);
        }
        
        reconData = rbm2000.getHidden(reconData);
        reconData = rbm2000.getVisible(reconData);
        
        float[][] reconDataFloat = reconData.toArray2();
        
        compareArraysForError(reconDataFloat, compareDataFloat, testName, true);
    }
    
    private static void dynamicReconstructionTest(RBM[] rbms, int edgeLength, float[][][] data, String testName, int numOfDreams, boolean resetOriginalPixels, boolean extraDream, boolean rgb) throws IOException{
        System.out.println("Starting Test: " + testName);
        
        float[][] compareDataFloat = data[1];
        FloatMatrix reconData = new FloatMatrix(data[0]);
        for(int j = 0; j < numOfDreams; ++j){
            System.out.println(testName + ", dream epoch: " + j);
            for(int i = 0; i < rbms.length; ++i){
                reconData = rbms[i].getHidden(reconData);
            }
            for(int i = rbms.length - 1; i >= 0; --i){
                reconData = rbms[i].getVisible(reconData);
            }
            
            if(resetOriginalPixels){
                float[][] reconDataFloat = reconData.toArray2();
                for(int i = 0; i < reconDataFloat.length; ++i){
                    replaceReconstructionWithOriginalPixels(reconDataFloat[i], compareDataFloat[i], 32, 64, 0, 64, 64, rgb);
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
        //contrastFix(reconDataFloat);
        
        System.out.println(compareDataFloat[0].length);
        System.out.println(reconDataFloat[0].length);
        
        compareArraysForError(reconDataFloat, compareDataFloat, testName, rgb);
    }
    
    private static void contrastFix(float[][] reconDataFloat){
        for(int i = 0; i < reconDataFloat.length; ++i){
            float lumRecon = luminaceMeanOfRegion(reconDataFloat[i], 0, 32, 0, 64, 64);
            float lumOriginal = luminaceMeanOfRegion(reconDataFloat[i], 32, 64, 0, 64, 64);
            float diff = lumOriginal - lumRecon;
            addLtoRegion(reconDataFloat[i], diff, 0, 32, 0, 64, 64);
            float lumReconNew = luminaceMeanOfRegion(reconDataFloat[i], 0, 32, 0, 64, 64);
            if(lumReconNew != lumOriginal) System.out.println("lumReconNew: " + lumReconNew + ", lumOriginal: " + lumOriginal);
            moreContrast(reconDataFloat[i], lumReconNew, 1.2f, 0, 32, 0, 64, 64);
        }
    }
    
    private static void moreContrast(float[] image, float L, float factor, int xStart, int xEnd, int yStart, int yEnd, int edgeLength){
        for(int i = yStart; i < yEnd; ++i){
            for(int j = xStart; j < xEnd; ++j){
                int pos = i * edgeLength + j;
                image[pos] = Math.max(0.0f, Math.min(1.0f, ((image[pos] - L) * factor + L)));
            }
        }
    }
    
    private static void replaceReconstructionWithOriginalPixels(float[] reconData, float[] originalData, int xStart, int xEnd, int yStart, int yEnd, int edgeLength, boolean rgb){
        if(rgb) replaceReconstructionWithRGB(reconData, originalData, xStart, xEnd, yStart, yEnd, edgeLength);
        else replaceReconstructionWithLuminance(reconData, originalData, xStart, xEnd, yStart, yEnd, edgeLength);
    }
    
    private static void replaceReconstructionWithRGB(float[] reconData, float[] originalData, int xStart, int xEnd, int yStart, int yEnd, int edgeLength){
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
    
    private static void replaceReconstructionWithLuminance(float[] reconData, float[] originalData, int xStart, int xEnd, int yStart, int yEnd, int edgeLength){
        for(int i = yStart; i < yEnd; ++i){
            int yPos = i;
            for(int j = xStart; j < xEnd; ++j){
                int xPos = j;
                int pos = yPos * edgeLength + xPos;
                reconData[pos] = originalData[pos];
            }
        }
    }
    
    private static float luminaceMeanOfRegion(float[] image, int xStart, int xEnd, int yStart, int yEnd, int edgeLength){
        float L = 0.0f;
        
        for(int i = yStart; i < yEnd; ++i){
            int yPos = i;
            for(int j = xStart; j < xEnd; ++j){
                int xPos = j;
                int pos = yPos * edgeLength + xPos;
                L += image[pos];
            }
        }
        int numOfPixels = (yEnd - yStart) * (xEnd - xStart);
        L /= numOfPixels;
        
        return L;
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
    
    private static void writeLtoRegion(float[] image, float L, int xStart, int xEnd, int yStart, int yEnd, int edgeLength){
        for(int i = yStart; i < yEnd; ++i){
            int yPos = i;
            for(int j = xStart; j < xEnd; ++j){
                int xPos = j;
                int pos = yPos * edgeLength + xPos;
                image[pos] = L;
            }
        }
    }
    
    private static void addLtoRegion(float[] image, float L, int xStart, int xEnd, int yStart, int yEnd, int edgeLength){
        for(int i = yStart; i < yEnd; ++i){
            int yPos = i;
            for(int j = xStart; j < xEnd; ++j){
                int xPos = j;
                int pos = yPos * edgeLength + xPos;
                image[pos] = Math.max(0.0f, Math.min(1.0f, image[pos] + L));
            }
        }
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
    
    private static void compareArraysForError(float[][] reconData, float[][] compareData, String testName, boolean rgb) throws IOException{
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
            
            BufferedImage bi = DataConverter.pixelDataToImage(reconData[i], 0.0f, rgb);
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
