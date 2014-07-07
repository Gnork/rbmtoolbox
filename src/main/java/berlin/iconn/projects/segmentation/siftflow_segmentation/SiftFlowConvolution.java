package berlin.iconn.projects.segmentation.siftflow_segmentation;

import berlin.iconn.projects.segmentation.SegmentationDataConverter;
import berlin.iconn.rbm.IRBM;

import java.io.File;
import java.util.Arrays;

/**
 * Created by Moritz on 6/30/2014.
 */
public class SiftFlowConvolution extends ASiftFlowDataProvider {


    private final IRBM rbm;

    public SiftFlowConvolution(int classesCount, int patchSize, float loadNewFilePosibility, int imageSize, File[] labelFiles, File[] imageFiles, IRBM rbm) {
        super(classesCount, patchSize, loadNewFilePosibility, imageSize, labelFiles, imageFiles);
        this.rbm = rbm;
        changeDataAtTraining();
    }

    @Override
    protected float[][] getMiniBatch(float[][] minibatch) {
        int hiddenLength = rbm.getWeights()[0].length - 1;
        int dataLength = hiddenLength * 9;
        int transpose = patchSize / 2;
        for (int i = 0; i < minibatch.length; i++) {
            float[] convolvePatch =  new float[dataLength];
            final int bound = imageSize - patchSize + 1;
            final int x = RANDOM.nextInt(bound);
            final int y = RANDOM.nextInt(bound);
            ImageWithLabels current = currentData[RANDOM.nextInt(currentData.length)];

            float[][] hidden = rbm.getHidden(new float[][]{SegmentationDataConverter.getPatchWithLabel(
                    x, y,
                    current.labels, current.image,
                    imageSize, patchSize, classesCount)});
             System.arraycopy(hidden[0], 0, convolvePatch, 0, hidden[0].length);
            int index = 1;
            for (int j = -1; j < 2; j++) {
                int xt = x + (j * transpose);
                for (int k = -1; k < 2; k++) {
                    if(i != 0 && j != 0) {
                        int yt = y + (i * transpose);
                        if(xt >= bound) xt = bound - 1;
                        if(xt < 0) xt = 0;
                        if(yt >= bound) yt = bound - 1;
                        if(yt < 0) yt = 0;
                        hidden = rbm.getHidden(new float[][]{SegmentationDataConverter.getPatchWithLabel(
                                xt, yt,
                                current.labels, current.image,
                                imageSize, patchSize, classesCount)});
                        System.arraycopy(hidden[0], 0, convolvePatch, index * hiddenLength, hidden[0].length);
                        index++;
                    }
                }
            }
            minibatch[i] = convolvePatch;
        }
        return minibatch;
    }
}
