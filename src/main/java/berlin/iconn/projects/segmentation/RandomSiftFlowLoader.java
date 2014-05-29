/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package berlin.iconn.projects.segmentation;

import berlin.iconn.persistence.InOutOperations;
import berlin.iconn.rbm.DataConverter;
import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;

/**
 *
 * @author christoph
 */
public class RandomSiftFlowLoader {
    private final Random random = new Random();
    
    private final File imagePath;
    private final File labelPath;
    private final int edgeLength;
    private final boolean binarize;
    private final boolean invert;
    private final float minData;
    private final float maxData;
    private final boolean isRGB;
    
    public RandomSiftFlowLoader(File imagePath, File labelPath, int edgeLength, boolean binarize, boolean invert, float minData, float maxData, boolean isRGB){
        this.imagePath = imagePath;
        this.labelPath = labelPath;
        this.edgeLength = edgeLength;
        this.binarize = binarize;
        this.invert = invert;
        this.minData = minData;
        this.maxData = maxData;
        this.isRGB = isRGB;
    }
    
    public void loadRandomImageAndLabels(float[][] imageDest, int[][] labelsDest){
        int count = imageDest.length;
        final File[] imageFiles = imagePath.listFiles((File dir, String name) -> (name.endsWith("jpg") || name.endsWith("png") || name.endsWith("gif")));
        final File[] labelFiles = labelPath.listFiles((File dir, String name) -> (name.endsWith("mat")));
        int fileLen = imageFiles.length;
        int nextRandom;
        for(int i = 0; i < count; ++i){
            nextRandom = random.nextInt(fileLen);
            try {
                imageDest[i] = DataConverter.processPixelData(ImageIO.read(imageFiles[nextRandom]), edgeLength, binarize, invert, minData, maxData, isRGB);
                labelsDest[i] = InOutOperations.loadSiftFlowLabel(labelFiles[i]);
            } catch (IOException ex) {
                Logger.getLogger(RandomSiftFlowLoader.class.getName()).log(Level.SEVERE, null, ex);
            }           
        }
    }    
}
