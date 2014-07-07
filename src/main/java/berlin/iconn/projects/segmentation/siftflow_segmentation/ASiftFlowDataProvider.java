package berlin.iconn.projects.segmentation.siftflow_segmentation;

import berlin.iconn.persistence.InOutOperations;
import berlin.iconn.projects.segmentation.SegmentationDataConverter;
import berlin.iconn.rbm.dataprovider.ATrainingDataProvider;
import org.jblas.FloatMatrix;

import java.io.File;
import java.io.IOException;
import java.util.Random;

/**
 * Created by Moritz on 6/30/2014.
 */
public abstract class ASiftFlowDataProvider extends ATrainingDataProvider {
    protected final static Random RANDOM = new Random();
    protected final File[] imageFiles;
    protected final File[] labelFiles;
    protected final int imageSize;
    protected final int patchSize;
    protected final int classesCount;
    protected final float loadNewFilePosibility;
    protected final ImageWithLabels[] currentData;
    protected final int miniBatchCount = 10;

    public ASiftFlowDataProvider(int classesCount, int patchSize, float loadNewFilePosibility, int imageSize, File[] labelFiles, File[] imageFiles) {
        super(new float[1][1]);
        this.classesCount = classesCount;
        this.patchSize = patchSize;
        this.currentData = new ImageWithLabels[50];
        this.loadNewFilePosibility = loadNewFilePosibility;
        this.imageSize = imageSize;
        this.labelFiles = labelFiles;
        this.imageFiles = imageFiles;
    }

    protected ImageWithLabels loadFromFiles(int index) {
        try {
         //   System.out.println("Loading: " + labelFiles[index].getName());
            return new ImageWithLabels( InOutOperations.loadSiftFlowLabel(imageSize, labelFiles[index]),
                    InOutOperations.getImageData(imageSize, false, false, 0.0f, 1.0f, true, imageFiles[index]).getData());
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

    protected abstract float[][] getMiniBatch(float[][] minibatch);

}
