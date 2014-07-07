package berlin.iconn.projects.segmentation.siftflow_segmentation;

/**
 * Created by Moritz on 7/2/2014.
 */

public class ImageWithLabels {
    public final int[] labels;
    public final float[] image;

    public ImageWithLabels(int[] labels, float[] image) {
        this.labels = labels;
        this.image = image;
    }
}
