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
    
    private static final String RBM1_WEIGHTS = "Output/SimpleWeights/WildFaces_64x64_rgb_1kh_2700it.dat";
    
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
        
        try {
            //reconstructionTest(rbms, 64, new File(IMAGES_TRAINED), new File(IMAGES_TRAINED), "IMAGES_TRAINED");
            reconstructionTest(rbms, 64, new File(IMAGES_TRAINED_INCOMPLETE), new File(IMAGES_TRAINED), "IMAGES_TRAINED_INCOMPLETE");
            //reconstructionTest(rbms, 64, new File(IMAGES_NOT_TRAINED), new File(IMAGES_NOT_TRAINED), "IMAGES_NOT_TRAINED");
            reconstructionTest(rbms, 64, new File(IMAGES_NOT_TRAINED_INCOMPLETE), new File(IMAGES_NOT_TRAINED), "IMAGES_NOT_TRAINED_INCOMPLETE");
        } catch (IOException ex) {
            Logger.getLogger(MainTest.class.getName()).log(Level.SEVERE, null, ex);
        }     
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
        for(int j = 0; j < 5; ++j){
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
