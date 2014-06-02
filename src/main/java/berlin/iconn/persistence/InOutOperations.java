/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package berlin.iconn.persistence;

import berlin.iconn.rbm.DataConverter;
import berlin.iconn.rbm.DataSet;

import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.file.Files;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.LinkedList;
import java.util.List;
import javax.imageio.ImageIO;
import org.apache.commons.io.FileUtils;

public class InOutOperations {
    private static final String simpleWeightsFolder = "Output/SimpleWeights";
    private static final String imageExportFolder = "Output/ImageExport"; 
    
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
    
    public static float[][] loadSimpleWeights(File file) throws IOException, ClassNotFoundException{
	ObjectInputStream ois = new ObjectInputStream(Files.newInputStream(file.toPath()));
        float[][] weights = (float[][]) ois.readObject();
        ois.close();
        return weights; 
    }
    
    public static void exportAsImage(float[] data, Date date, String suffix) throws IOException {
        exportAsImage(new float[][]{data}, date, suffix);
    }
    
    public static void exportAsImage(float[][] data, Date date, String suffix) throws IOException {
        String path = imageExportFolder + "/" + getDirectoryNameByDate(date, suffix);
        mkdir(path);
        for (int i = 0; i < data.length; i++) {
            BufferedImage image = DataConverter.pixelDataToImage(data[i], 0.0f, false);
            File file = new File(path + "/" + i + ".png");
            ImageIO.write(image, "png", file);
        }
    }
    
    public static DataSet[] loadImages(File path, int edgeLength, int padding, boolean binarize, boolean invert, float minData, float maxData, boolean isRGB) throws IOException {

        final File[] imageFiles = path.listFiles(new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                return (name.endsWith("jpg") || name.endsWith("png") || name.endsWith("gif"));
            }
        });

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
    
    public static int[][] loadSiftFlowLabels(File path) throws FileNotFoundException, IOException{
        final File[] labelsFiles = path.listFiles((File dir, String name) -> (name.endsWith(".mat")));
        
        int[][] result = new int[labelsFiles.length][];
        
        for(int i = 0; i < labelsFiles.length; ++i){
            result[i] = loadSiftFlowLabel(labelsFiles[i]);
        }
        return result;
    }
    
    public static int[] loadSiftFlowLabel(File file) throws FileNotFoundException, IOException{
        int n = 256;
        int[] result = new int[n * n];
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            for(int j = 0; j < 5; ++j){
                br.readLine();
            }
            String line;
            int lineCount = 0;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if(! line.equals("")){
                    result[lineCount] = new Integer(line);
                }
                ++lineCount;
            }
        }
        
        // transpose
        int index1, index2, temp;
        for (int i = 0; i < n; i++){
            for (int j = i + 1; j < n;  j++){
               index1 = i * n + j;
               index2 = j * n + i;
               temp = result[index1];
               result[index1] = result[index2];
               result[index2] = temp;
            }
        }
        return result;
    }
    
    public static String[] loadSiftFlowClasses(File file) throws FileNotFoundException, IOException{
        List<String> resultList = new LinkedList<>();
        resultList.add("undefined");
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = br.readLine()) != null) {
                if(! line.contains("#")){
                    line = line.trim();
                    if(! line.isEmpty()){
                        resultList.add(line); 
                    }
                }
            }
        }
        String[] result = new String[resultList.size()];
        resultList.toArray(result);
        return result;
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
    
    private static final void mkdir(String path) throws IOException{
        File file = new File(path);
        if(!file.isDirectory()){
            FileUtils.forceMkdir(file);
        }           
    }
}
