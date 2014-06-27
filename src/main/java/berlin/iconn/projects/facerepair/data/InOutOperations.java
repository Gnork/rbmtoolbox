package berlin.iconn.projects.facerepair.data;

import java.io.File;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.file.Files;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;
import org.apache.commons.io.FileUtils;

/**
 * DataConverter
 *
 * @author Radek
 */
public class InOutOperations {
    private static final String simpleWeightsFolder = "Output/SimpleWeights";
    private static final String imageExportFolder = "Output/ImageExport"; 
    
    public static DataSet[] loadImages(File path, int edgeLength, int padding, boolean binarize, boolean invert, float minData, float maxData, boolean isRGB) throws IOException {

        final File[] imageFiles = path.listFiles((File dir, String name) -> (name.endsWith("jpg") || name.endsWith("png") || name.endsWith("gif")));

        int size = edgeLength * edgeLength;
        DataSet[] result = new DataSet[imageFiles.length];

        for (int i = 0; i < imageFiles.length; i++) {
            float[] imageData;
            imageData = DataConverter.processPixelData(ImageIO.read(imageFiles[i]), edgeLength, binarize, invert, minData, maxData, isRGB);
            // Pad does not work with RGB images
            //imageData = DataConverter.pad(imageData, edgeLength, padding);

            String label = imageFiles[i].getName().split("_")[0];
            result[i] = new DataSet(imageData, label);
        }

        return result;
    }
    
    public static void saveSimpleWeights(float[][] weights, Date date) throws IOException{
        saveSimpleWeights(weights, date, "weights");
    }
    
    public static void saveSimpleWeights(float[][] weights, Date date, String suffix) throws IOException{
        mkdir(simpleWeightsFolder);
        File file = new File(simpleWeightsFolder + "/" + getFileNameByDate(date, suffix, "dat"));
        ObjectOutputStream oos = new ObjectOutputStream(Files.newOutputStream(file.toPath()));
        oos.writeObject(weights);
        oos.close();
    }
    
    public static float[][] loadSimpleWeights(String path){
        float[][] weights = null;
        
        ObjectInputStream ois = null;
        try {
            File file = new File(path);
            ois = new ObjectInputStream(Files.newInputStream(file.toPath()));
            weights = (float[][]) ois.readObject();
            ois.close(); 
        } catch (IOException | ClassNotFoundException ex) {
            Logger.getLogger(InOutOperations.class.getName()).log(Level.SEVERE, null, ex);
        } finally {
            try {
                ois.close();
            } catch (IOException ex) {
                Logger.getLogger(InOutOperations.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        
        return weights;
    }
    
    private static String getFileNameByDate(Date date, String suffix, String extension){
        if(date == null){
            return null;
        }
        String result = getDirectoryNameByDate(date, suffix);       
        extension = extension.trim();
        extension = extension.replaceAll(" ", "_");
        if(! extension.isEmpty()){
            if(! extension.startsWith(".")){
                result += ".";
            }
            result += extension;
        }
        return result;
    }
    
    private static String getDirectoryNameByDate(Date date, String suffix){
        if(date == null){
            return null;
        }
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy_MM_dd_HH_mm_ss");
        String result = sdf.format(date);
        suffix = suffix.trim();
        suffix = suffix.replaceAll("\\.", "");
        suffix = suffix.replaceAll(" ", "_");
        if(! suffix.isEmpty()){
            result += "_" + suffix;
        }
        return result;
    }
    
    public static final void mkdir(String path) throws IOException{
        File file = new File(path);
        if(!file.isDirectory()){
            FileUtils.forceMkdir(file);
        }           
    }
}
