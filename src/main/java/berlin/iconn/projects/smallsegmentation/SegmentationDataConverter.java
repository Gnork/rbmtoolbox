package berlin.iconn.projects.smallsegmentation;

import org.jblas.FloatMatrix;

import java.awt.image.BufferedImage;
import java.util.ArrayList;

/**
 * Created by Moritz on 6/5/2014.
 */
public class SegmentationDataConverter {


    public static FloatMatrix[] createTrainingData(int[] labels, float[] image, int width, int height, int patchSize, int classLength) {
        FloatMatrix[] data = new FloatMatrix[2];
        int patchSizeMin1 = patchSize - 1;
        final int dataLength = labels.length - (patchSizeMin1 * patchSizeMin1);
        float[][] labelData = new float[dataLength][classLength];
        float[][] imageData = new float[dataLength][patchSize * patchSize * 3];

        int low = patchSize / 2;
        final int newWidth = width - patchSizeMin1;
        for (int i = 0; i < height - patchSizeMin1; i++) {
            int labelY = (i + low) * width + low;
            int dataY = i * newWidth;
            for (int j = 0; j < newWidth; j++) {
                final int dataPos = dataY + j;
                labelData[dataPos][labels[labelY + j]] = 1.0f;
                for (int k = 0; k < patchSize; k++) {
                    for (int a = 0; a < patchSize; a++) {
                        int imagePos = ((i + k) * width + j + a) * 3;
                        int patchPos = (k * patchSize + a) * 3;
                        imageData[dataPos][patchPos] = image[imagePos];
                        imageData[dataPos][patchPos + 1] = image[imagePos + 1];
                        imageData[dataPos][patchPos + 2] = image[imagePos + 2];
                    }
                }
            }
        }
        data[0] = new FloatMatrix(labelData);
        data[1] = new FloatMatrix(imageData);
        return data;
    }


    public static int[] getImageData(float[][] data, int width, int height, int patchSize) {
        int[] result = new int[width * height];
        int low = patchSize / 2;
        int patchSizeMin1 = patchSize - 1;
        final int newWidth = width - patchSizeMin1;
        for (int i = 0; i < height - patchSizeMin1; i += patchSize) {
            for (int j = 0; j < newWidth; j += patchSize) {
                final int dataPos = i * newWidth + j;
                for (int k = 0; k < patchSize; k++) {
                    int patchY = k * patchSize;
                    for (int a = 0; a < patchSize; a++) {
                        int patchPos = (patchY + a) * 3;
                        int r = (int) (data[dataPos][patchPos] * 255);
                        patchPos++;
                        int g = (int) (data[dataPos][patchPos] * 255);
                        patchPos++;
                        int b = (int) (data[dataPos][patchPos] * 255);
                        int pos = (i + k) * width + j + a;
                        result[pos] = (r << 16) | (g << 8) | b;
                    }
                }
            }
        }
        return result;
    }

    public static int[] getLabelData(float[][] data, int width, int height, int patchSize) {
        int[] result = new int[width * height];
        int low = patchSize / 2;
        int patchSizeMin1 = patchSize - 1;
        final int newWidth = width - patchSizeMin1;

        for (int i = 0; i < height - patchSizeMin1; i++) {
            for (int j = 0; j < newWidth; j++) {
                float max = Float.NEGATIVE_INFINITY;
                int index = 0;
                final int posLabel = i * newWidth + j;
                for (int k = 0; k < data[0].length; k++) {
                    if (data[posLabel][k] > max) {
                        index = k;
                        max = data[posLabel][k];
                    }
                }
                result[(i + low) * width + j + low] = index;
            }
        }
        return result;
    }
}
