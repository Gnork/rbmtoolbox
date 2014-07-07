package berlin.iconn.projects.segmentation.siftflow_segmentation;

import berlin.iconn.persistence.InOutOperations;
import berlin.iconn.projects.segmentation.SegmentationDataConverter;
import org.jblas.FloatMatrix;

import java.io.File;
import java.io.IOException;

/**
 * Created by Moritz on 7/2/2014.
 */
public class SiftFlowDataProviderWithMiniImage extends ASiftFlowDataProvider {
    private final float[][] miniImages;
    private final int miniImageSize;
    private final int unfoldMiniImageSize;

    public SiftFlowDataProviderWithMiniImage(int classesCount, int patchSize, float loadNewFilePosibility,
                                             int imageSize, int miniImageSize, File[] labelFiles, File[] imageFiles) {
        super(classesCount, patchSize, loadNewFilePosibility, imageSize, labelFiles, imageFiles);
        this.miniImageSize = miniImageSize;
        this.unfoldMiniImageSize = miniImageSize * miniImageSize * 3;
        miniImages = new float[currentData.length][];
        for (int i = 0; i < currentData.length; i++) {
            final int index = RANDOM.nextInt(imageFiles.length);
            this.currentData[i] = loadFromFiles(index);
            getMiniImage(i, index);
        }
        changeDataAtTraining();
    }
    @Override
    public void changeDataAtTraining() {
        setDataWithBias(null);
        if(RANDOM.nextFloat() < loadNewFilePosibility) {
            final int currentDataIndex = RANDOM.nextInt(currentData.length);
            final int index = RANDOM.nextInt(imageFiles.length);
            currentData[currentDataIndex] = loadFromFiles(index);
            getMiniImage(currentDataIndex, index);
        }

        setData(new FloatMatrix(getMiniBatch(new float[miniBatchCount][])));
    }

    private void getMiniImage(int currentDataIndex, int fileIndex) {
        try {
            miniImages[currentDataIndex] = InOutOperations.getImageData(miniImageSize,
                    false, false, 0.0f, 1.0f, true, imageFiles[fileIndex]).getData();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    protected float[][] getMiniBatch(float[][] minibatch) {
        for (int i = 0; i < miniBatchCount; i++) {
            int x = RANDOM.nextInt(imageSize - patchSize + 1);
            int y = RANDOM.nextInt(imageSize - patchSize + 1);
            int index = RANDOM.nextInt(currentData.length);
            ImageWithLabels pic = currentData[index];
            float[] patchWithLabel = SegmentationDataConverter.getPatchWithLabel(x, y, pic.labels, pic.image, imageSize, patchSize, classesCount);
            float[] totalData = new float[patchWithLabel.length + unfoldMiniImageSize];
            System.arraycopy(patchWithLabel, 0, totalData, 0, patchWithLabel.length);
            System.arraycopy(miniImages[index], 0, totalData, patchWithLabel.length, unfoldMiniImageSize);
            minibatch[i] = totalData;
        }
        return minibatch;
    }


}
