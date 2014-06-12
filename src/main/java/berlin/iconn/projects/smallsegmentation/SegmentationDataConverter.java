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
        ArrayList<float[]> labelData = new ArrayList<>();
        ArrayList<float[]> imageData = new ArrayList<>();

        int low = patchSize / 2;
        int high = (patchSize % 2 == 0) ? low : low + 1;

        for (int i = low; i < height - high; i++) {
            int labelY = i * width;
            for (int j = low; j < width - high; j++) {
                float[] label = new float[classLength];
                label[labels[labelY + j]] = 1.0f;
                labelData.add(label);
                float[] patch = new float[patchSize * patchSize * 3];
                int index = 0;
                for (int k = -low; k < high; k++) {
                    int imageY = (i + k) * width * 3;
                    for (int a = -low; a < high; a++) {
                        int pos = imageY + (j + a) * 3;
                        patch[index] = image[pos];
                        index++;
                        pos++;
                        patch[index] = image[pos];
                        index++;
                        pos++;
                        patch[index] = image[pos];
                        index++;
                    }
                }
                imageData.add(patch);
            }
        }
        data[0] = new FloatMatrix(labelData.toArray(new float[0][]));
        data[1] = new FloatMatrix(imageData.toArray(new float[0][]));
        return data;
    }

    public static int[] getLabelData(float[][] data, int width, int height, int patchSize) {
        int[] result = new int[width * height];
        int low = patchSize / 2;
        int high = (patchSize % 2 == 0) ? low : low + 1;
        final int newWidth = width - high;
        for (int i = low; i < height - high; i++) {
            for (int j = low; j < newWidth; j++) {
                float max = Float.NEGATIVE_INFINITY;
                int index = 0;
                final int posLabel = (i - low) * (newWidth - low) + (j - low);
                for (int k = 0; k < data[0].length; k++) {
                    if (data[posLabel][k] > max) {
                        index = k;
                        max = data[posLabel][k];
                    }
                }
                result[i * width + j] = index;
            }
        }
        return result;
    }

    public static BufferedImage getImageData(float[][] data, int width, int height, int patchSize) {
        int[] result = new int[width * height];
        int low = patchSize / 2;
        int high = (patchSize % 2 == 0) ? low : low + 1;
        int index = 0;
        for (int i = low; i < height - high; i += patchSize) {
            for (int j = low; j < width - high; j += patchSize) {

                for (int k = -low; k < high; k++) {
                    int patchY = (k + low) * patchSize * 3;
                    for (int h = -low; h < high; h++) {
                        int patchPos = patchY + (h + low) * 3;
                        int r = (int) (data[index][patchPos] * 255);
                        patchPos++;
                        int g = (int) (data[index][patchPos] * 255);
                        patchPos++;
                        int b = (int) (data[index][patchPos] * 255);
                        int pos = (i + k) * width + j + h;
                        result[pos] = 0xFF000000 | (r << 16) | (g << 8) | b;
                    }
                }
                index++;
            }
        }
        BufferedImage bufferedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        bufferedImage.setRGB(0, 0, width, height, result, 0, width);
        return bufferedImage;
    }
}
