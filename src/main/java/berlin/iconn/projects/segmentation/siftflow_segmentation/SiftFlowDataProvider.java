package berlin.iconn.projects.segmentation.siftflow_segmentation;

import berlin.iconn.persistence.InOutOperations;
import berlin.iconn.projects.segmentation.SegmentationDataConverter;
import berlin.iconn.rbm.dataprovider.ATrainingDataProvider;
import org.jblas.FloatMatrix;

import java.io.File;
import java.io.IOException;
import java.util.LinkedList;
import java.util.Random;

/**
 * Created by Moritz on 6/25/2014.
 */
public class SiftFlowDataProvider extends ASiftFlowDataProvider {

    public SiftFlowDataProvider(File[] imageFiles, File[] labelFiles,int imageSize, int patchSize, int classesCount, float loadNewFilePosibility) {
        super(classesCount, patchSize, loadNewFilePosibility, imageSize, labelFiles, imageFiles);
        for (int i = 0; i < currentData.length; i++) {
            this.currentData[i] = loadFromFiles(RANDOM.nextInt(imageFiles.length));
        }

        changeDataAtTraining();
    }


    protected float[][] getMiniBatch(float[][] minibatch) {
        for (int i = 0; i < miniBatchCount; i++) {
            int x = RANDOM.nextInt(imageSize - patchSize + 1);
            int y = RANDOM.nextInt(imageSize - patchSize + 1);
            ImageWithLabels pic = currentData[RANDOM.nextInt(currentData.length)];
            minibatch[i] = SegmentationDataConverter.getPatchWithLabel(x, y, pic.labels, pic.image, imageSize, patchSize, classesCount);
        }
        return minibatch;
    }
}
