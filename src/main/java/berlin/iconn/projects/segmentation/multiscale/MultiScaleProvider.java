package berlin.iconn.projects.segmentation.multiscale;

import berlin.iconn.persistence.InOutOperations;
import berlin.iconn.rbm.IRBM;
import berlin.iconn.rbm.dataprovider.ATrainingDataProvider;
import org.jblas.FloatMatrix;

import java.io.File;
import java.io.IOException;
import java.util.Random;

/**
 * Created by Moritz on 7/11/2014.
 */
public class MultiScaleProvider extends ATrainingDataProvider {

    protected final static Random RANDOM = new Random();
    protected final File[] imageFiles;
    protected final File[] labelFiles;
    private final int[] imageSizes;
    protected final int patchSize;
    protected final int classesCount;
    protected final float loadNewFilePosibility;
    protected final MultiImageWithLabels[] currentData;
    protected final IRBM[] rbms;
    protected final int miniBatchCount = 10;
    private final int patchSizeHalf;

    public MultiScaleProvider(float loadNewFilePosibility, File[] imageFiles, File[] labelFiles, IRBM[] rbms, int[] imageSizes, int classesCount) {
        this.loadNewFilePosibility = loadNewFilePosibility;
        this.imageFiles = imageFiles;
        this.labelFiles = labelFiles;
        this.imageSizes = imageSizes;
        this.patchSize = imageSizes[imageSizes.length - 1];
        this.patchSizeHalf = patchSize / 2;
        this.classesCount = classesCount;
        this.rbms = rbms;
        this.currentData = new MultiImageWithLabels[50];

        for (int i = 0; i < currentData.length; i++) {
            this.currentData[i] = loadFromFiles(RANDOM.nextInt(imageFiles.length));
        }

        changeDataAtTraining();
    }
    public MultiImageWithLabels loadFromFiles(File labelFile, File imageFile) {
        int[] labels = null;
        float[][] images = new float[getImageSizes().length][];
        try {
            labels = InOutOperations.loadSiftFlowLabel(getImageSizes()[getImageSizes().length - 2], labelFile);
        } catch (IOException e) {
            e.printStackTrace();
        }
        for (int i = 0; i < getImageSizes().length; i++) {
            try {
                images[i] = InOutOperations.getImageData(getImageSizes()[i], false, false, 0.0f, 1.0f, true, imageFile).getData();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return new MultiImageWithLabels(labels, images);
    }

    protected MultiImageWithLabels loadFromFiles(int index) {
        return loadFromFiles(labelFiles[index], imageFiles[index]);
    }
    @Override
    public void changeDataAtTraining() {

        setDataWithBias(null);
        if(RANDOM.nextFloat() < loadNewFilePosibility) {
            currentData[RANDOM.nextInt(currentData.length)] = loadFromFiles(RANDOM.nextInt(imageFiles.length));
        }

        setData(new FloatMatrix(getMiniBatch(new float[miniBatchCount][])));
    }

    private float[][] getMiniBatch(float[][] minibatch) {
        for (int i = 0; i < miniBatchCount; i++) {
            int x = RANDOM.nextInt(getImageSizes()[getImageSizes().length - 2] - patchSize + 1);
            int y = RANDOM.nextInt(getImageSizes()[getImageSizes().length - 2] - patchSize + 1);
            MultiImageWithLabels pic = currentData[RANDOM.nextInt(currentData.length)];
            minibatch[i] = getFeatures(pic, x, y, true);
        }

        return minibatch;
    }

    public float[] getFeatures(MultiImageWithLabels pic, int x, int y, boolean addlabel) {
        int rbmHidden = rbms[rbms.length - 1].getWeights()[0].length - 1;
        int indexOfSmallPictureSize = getImageSizes().length - 2;
        int smallImageSize = getImageSizes()[indexOfSmallPictureSize];

        float[] result = new float[classesCount + getImageSizes().length * rbmHidden];

        if(addlabel) {
            result[pic.labels[(y + patchSizeHalf) * smallImageSize + x + patchSizeHalf]] = 1.0f;
        }
        for (int i = 0; i < indexOfSmallPictureSize; i++) {
            int positionScale = getImageSizes()[i] / smallImageSize;
            int xt = x * positionScale;
            int yt = y * positionScale;

            float[] image = pic.images[i];
            float[] patch = new float[patchSize * patchSize * 3];
            for (int j = 0; j < patchSize; j++) {
                for (int k = 0; k < patchSize; k++) {
                    int imagePos = ((yt + j) * getImageSizes()[i] + xt + k) * 3;
                    int patchPos = (j * patchSize + k) * 3;
                    patch[patchPos] = image[imagePos];
                    patch[patchPos + 1] = image[imagePos + 1];
                    patch[patchPos + 2] = image[imagePos + 2];
                }
            }
            patch = runHidden(patch);
            System.arraycopy(patch, 0, result, classesCount + i * rbmHidden, rbmHidden);
        }

        System.arraycopy(runHidden(pic.images[getImageSizes().length - 1]), 0,
                result, classesCount + (getImageSizes().length - 1) * rbmHidden, rbmHidden);
        return result;
    }

    private float[] runHidden(float[] patch) {
        float[][] hidden = new float[][]{ patch };
        for (int i = 0; i < rbms.length; i++) {
            hidden = rbms[i].getHidden(hidden);
        }
        return hidden[0];
    }

    public int[] getImageSizes() {
        return imageSizes;
    }
}
