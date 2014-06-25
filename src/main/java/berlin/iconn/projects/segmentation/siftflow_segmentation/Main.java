package berlin.iconn.projects.segmentation.siftflow_segmentation;

/**
 * Created by Moritz on 6/25/2014.
 */
public class Main {

    public static void main(String[] args) {
        new SiftFlowDataProvider(
                "Data/SiftFlowDataset/Images/spatial_envelope_256x256_static_8outdoorcategories",
                "Data/SiftFlowDataset/SemanticLabels/labels");
    }
}
