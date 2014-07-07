package berlin.iconn.rbm.dataprovider;

import berlin.iconn.persistence.InOutOperations;
import berlin.iconn.projects.segmentation.SegmentationDataConverter;
import berlin.iconn.projects.segmentation.siftflow_segmentation.ImageWithLabels;
import org.jblas.FloatMatrix;

import java.io.File;
import java.io.IOException;
import java.util.Random;

/**
 * Created by Moritz on 6/30/2014.
 */
public class RandomPoolPatchDataProvider extends ATrainingDataProvider {
    protected final static Random RANDOM = new Random();
    protected final File[] imageFiles;
    protected final int imageSize;
    protected final int patchSize;
    protected final float loadNewFilePosibility;
    protected final float[][] currentData;
    protected final int miniBatchCount = 10;


    public RandomPoolPatchDataProvider(int patchSize, float loadNewFilePosibility, int imageSize, File[] imageFiles) {
        super(new float[1][1]);
        this.patchSize = patchSize;
        this.currentData = new float[50][];
        this.loadNewFilePosibility = loadNewFilePosibility;
        this.imageSize = imageSize;
        this.imageFiles = imageFiles;
        for (int i = 0; i < currentData.length; i++) {
            this.currentData[i] = loadFromFiles(RANDOM.nextInt(imageFiles.length));
        }
        changeDataAtTraining();
    }

    protected float[] loadFromFiles(int index) {
        try {
            return InOutOperations.getImageData(imageSize, false, false, 0.0f, 1.0f, true, imageFiles[index]).getData();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    @Override
    public void changeDataAtTraining() {
        setDataWithBias(null);
        if(RANDOM.nextFloat() < loadNewFilePosibility) {
            currentData[RANDOM.nextInt(currentData.length)] = loadFromFiles(RANDOM.nextInt(imageFiles.length));
        }

        setData(new FloatMatrix(getMiniBatch(new float[miniBatchCount][])));
    }

    protected float[][] getMiniBatch(float[][] minibatch) {
        int patchSizeRGB = patchSize * 3;
        for (int i = 0; i < miniBatchCount; i++) {
            int x = RANDOM.nextInt(imageSize - patchSize + 1) * 3;
            int y = RANDOM.nextInt(imageSize - patchSize + 1);
            float[] pic = currentData[RANDOM.nextInt(currentData.length)];
            float[] patch = new float[patchSize * patchSizeRGB];
            for (int j = 0; j < patchSize; j++) {
                for (int k = 0; k < patchSizeRGB; k++) {
                    patch[j * patchSizeRGB + k] = pic[((y + j) * imageSize + x + k)];
                }
            }
            minibatch[i] = patch;
        }
        return minibatch;
    }

}
