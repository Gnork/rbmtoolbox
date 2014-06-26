package berlin.iconn.projects.facerepair.data;

/**
 * Created by Moritz on 5/18/2014.
 */
public class DataSet {
    private final float[] data;
    private final String label;

    public DataSet(float[] data, String label){
        this.data = data;
        this.label = label;
    }

    public float[] getData(){
        return data;
    }

    public String getLabel(){
        return label;
    }
}