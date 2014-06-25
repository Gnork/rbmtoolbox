package berlin.iconn.projects.segmentation.siftflow_segmentation;

import berlin.iconn.persistence.InOutOperations;
import berlin.iconn.rbm.dataprovider.ATrainingDataProvider;
import org.apache.commons.io.FilenameUtils;

import java.io.File;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.Queue;
import java.util.stream.Stream;

/**
 * Created by Moritz on 6/25/2014.
 */
public class SiftFlowDataProvider  extends ATrainingDataProvider{

    private final File[] imagePaths;
    private final String siftFlowLabelPath;
    private final LinkedList<ImageWithLabels> currentData;

    public SiftFlowDataProvider(String siftFlowImagePath, String sifFlowLabelPath) {
        super(new float[1][1]);
        this.siftFlowLabelPath = sifFlowLabelPath;

        imagePaths = InOutOperations.getImageFiles(new File(siftFlowImagePath));
        for (File file : imagePaths) {
            System.out.println(getLabelFile(file));
        }

        this.currentData = new LinkedList<>();
    }

    private File getLabelFile(File imageFile) {
        String labelName = FilenameUtils.removeExtension(imageFile.getName()) + ".mat";
        return new File(new File(siftFlowLabelPath).getAbsoluteFile() + "/" + labelName);
    }

    @Override
    public void changeDataAtTraining() {

    }

    private class ImageWithLabels {
    }
}
