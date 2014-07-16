package berlin.iconn.rbm.dataprovider;

import berlin.iconn.persistence.InOutOperations;
import org.jblas.FloatMatrix;

import java.io.File;
import java.io.IOException;
import java.util.Random;

/**
 * Created by Moritz on 6/30/2014.
 */
public class RandomPoolDataProvider extends ATrainingDataProvider {
    protected final static Random RANDOM = new Random();
    protected final File[] imageFiles;
    protected final int imageSize;
    protected float[][] currentData;
    protected final int miniBatchCount = 10;
    private float loadNewFilePosibility = 0.1f;


    public RandomPoolDataProvider(int imageSize, File[] imageFiles) {
        super(new float[1][1]);
        this.imageSize = imageSize;
        this.imageFiles = imageFiles;

        this.currentData = new float[200][];
        for (int i = 0; i < currentData.length; i++) {
            currentData[i] = loadFromFiles(RANDOM.nextInt(imageFiles.length));
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
            for (int i = 0; i < currentData.length / 10; i++) {
                currentData[RANDOM.nextInt(currentData.length)] = loadFromFiles(RANDOM.nextInt(imageFiles.length));
            }
        }
        float[][] minibatch = new float[miniBatchCount][];
        for (int i = 0; i < miniBatchCount; i++) {
            minibatch[i] = currentData[RANDOM.nextInt(currentData.length)];
        }

        setData(new FloatMatrix(minibatch));
    }




}
