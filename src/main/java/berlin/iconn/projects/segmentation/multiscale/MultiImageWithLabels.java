package berlin.iconn.projects.segmentation.multiscale;

/**
 * Created by Moritz on 7/13/2014.
 */
public class MultiImageWithLabels {
    public final int[] labels;
    public final float[][] images;

    public MultiImageWithLabels(int[] labels, float[][] images) {
        this.labels = labels;
        this.images = images;
    }
}
