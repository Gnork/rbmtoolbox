/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package berlin.iconn.persistence;

import berlin.iconn.rbm.DataConverter;
import berlin.iconn.rbm.DataSet;
import berlin.iconn.rbm.IRBM;

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
        mkdir(simpleWeightsFolder);
        File file = new File(simpleWeightsFolder + "/" + getFileNameByDate(date, "weights", "dat"));
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
    
    public static void exportAsImage(float[] data, Date date, String prefix) throws IOException {
        exportAsImage(new float[][]{data}, date, prefix);
    }
    
    public static void exportAsImage(float[][] data, Date date, String prefix) throws IOException {
        String path = imageExportFolder + "/" + getDirectoryNameByDate(date, prefix);
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
            imageData = DataConverter.pad(imageData, edgeLength, padding);

            String label = imageFiles[i].getName().split("_")[0];
            result[i] = new DataSet(imageData, label);
        }

        return result;
    }
    
    public static int[][] loadSiftFlowLabels(File path) throws FileNotFoundException, IOException{
        final File[] labelsFiles = path.listFiles((File dir, String name) -> (name.endsWith(".mat")));
        
        int[][] result = new int[labelsFiles.length][256 * 256];
        
        for(int i = 0; i < labelsFiles.length; ++i){
            try (BufferedReader br = new BufferedReader(new FileReader(labelsFiles[i]))) {
                for(int j = 0; j < 5; ++j){
                    br.readLine();
                }
                String line;
                int lineCount = 0;
                while ((line = br.readLine()) != null) {
                    line = line.trim();
                    if(! line.equals("")){
                        result[i][lineCount] = new Integer(line);
                    }
                    ++lineCount;
                }
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
    
    private static String getFileNameByDate(Date date, String prefix, String extension){
        if(date == null){
            return null;
        }
        String result = getDirectoryNameByDate(date, prefix);       
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
    
    private static String getDirectoryNameByDate(Date date, String prefix){
        if(date == null){
            return null;
        }
        SimpleDateFormat sdf = new SimpleDateFormat("dd_MM_yyyy_HH_mm_ss");
        String result = "";
        prefix = prefix.trim();
        prefix = prefix.replaceAll("\\.", "");
        prefix = prefix.replaceAll(" ", "_");
        System.out.println("Prefix: " + prefix);
        if(! prefix.isEmpty()){
            result += prefix + "_";
        }
        result += sdf.format(date);
        return result;
    }
    
    private static final void mkdir(String path) throws IOException{
        File file = new File(path);
        if(!file.isDirectory()){
            FileUtils.forceMkdir(file);
        }           
    }
}
