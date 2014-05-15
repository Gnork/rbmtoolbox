/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package berlin.iconn.persistence;

import java.io.File;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.file.Files;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.commons.io.FileUtils;

public class Persistor {
    private final String simpleWeightsFolder = "SimpleWeights";
    
    private final Date date; 
    
    public Persistor(){
        date = new Date();
        try {
            mkdir(simpleWeightsFolder);
        } catch (IOException ex) {
            Logger.getLogger(Persistor.class.getName()).log(Level.SEVERE, null, ex);
        }    
    }
    
    public static final void mkdir(String path) throws IOException{
        File file = new File(path);
        if(!file.isDirectory()){
                FileUtils.forceMkdir(file);
        }
            
    }
    
    public static void saveSimpleWeights(float[][] weights, File file) throws IOException{
        ObjectOutputStream oos = new ObjectOutputStream(Files.newOutputStream(file.toPath()));
        oos.writeObject(weights);
        oos.close();
    }
    
    public void overwriteSimpleWeights(float[][] weights) throws IOException{
        File file = new File(simpleWeightsFolder + "/" + getFileNameByDate(date, "weights", "dat"));
        saveSimpleWeights(weights, file);        
    }
    
    public void saveSimpleWeights(float[][] weights) throws IOException{
        File file = new File(simpleWeightsFolder + "/" + getFileNameByDate(new Date(), "weights", "dat"));
        saveSimpleWeights(weights, file);
    }
    
    public static String getFileNameByDate(Date date, String prefix, String extension){
        if(date == null){
            return null;
        }
        SimpleDateFormat sdf = new SimpleDateFormat("dd_MM_yyyy_HH_mm_ss");
        String result = "";
        prefix = prefix.trim();
        extension = extension.trim();
        prefix = prefix.replaceAll(".", "");
        prefix = prefix.replaceAll(" ", "_");
        extension = extension.replaceAll(" ", "_");
        if(! prefix.isEmpty()){
            result += prefix + "_";
        }
        result += sdf.format(date);
        if(! extension.isEmpty()){
            if(! extension.startsWith(".")){
                result += ".";
            }
            result += extension;
        }
        return result;
    }
    
    private static float[][] loadSimpleWeights(File file) throws IOException, ClassNotFoundException{
	ObjectInputStream ois = new ObjectInputStream(Files.newInputStream(file.toPath()));
        float[][] weights = (float[][]) ois.readObject();
        ois.close();
        return weights; 
    }
}
