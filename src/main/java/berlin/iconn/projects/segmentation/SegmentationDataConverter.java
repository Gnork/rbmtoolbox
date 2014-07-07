package berlin.iconn.projects.segmentation;

import org.jblas.FloatMatrix;

/**
 * Created by Moritz on 6/5/2014.
 */
public class SegmentationDataConverter {


    public static FloatMatrix[] createTrainingData(int[] labels, float[] image, int edgeLength, int patchSize, int classLength) {
        FloatMatrix[] data = new FloatMatrix[2];
        int patchSizeMin1 = patchSize - 1;
        final int dataLength = labels.length - (patchSizeMin1 * patchSizeMin1);
        float[][] labelData = new float[dataLength][classLength];
        float[][] imageData = new float[dataLength][patchSize * patchSize * 3];

        int low = patchSize / 2;
        final int newWidth = edgeLength - patchSizeMin1;
        for (int i = 0; i < edgeLength - patchSizeMin1; i++) {
            int labelY = (i + low) * edgeLength + low;
            int dataY = i * newWidth;
            for (int j = 0; j < newWidth; j++) {
                final int dataPos = dataY + j;
                final int classNumber = labels[labelY + j];
                if(classNumber >= classLength) System.out.println(classNumber);
                labelData[dataPos][classNumber] = 1.0f;
                for (int k = 0; k < patchSize; k++) {
                    for (int a = 0; a < patchSize; a++) {
                        int imagePos = ((i + k) * edgeLength + j + a) * 3;
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

    public static float[][] createPackedLabelData(int[]labels, int classesCount) {
        int packedBitSize = (int)Math.ceil(Math.log(classesCount) / Math.log(2));
        float[][] result = new float[labels.length][packedBitSize];
        for (int i = 0; i < labels.length; i++) {
            int classtype = labels[i];
            for (int j = 0; j < packedBitSize; j++) {
                result[i][j] = (classtype & 1) == 1 ? 1 : 0;
                classtype >>= 1;
            }
        }
        return result;
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



    public static float[] getPatchWithLabel(int x, int y, int[] labels, float[] image, int imageSize, int patchSize, int classesCount) {
        float[] result = new float[patchSize * patchSize * 3 + classesCount];

        int low = patchSize / 2;
        int labelY = (y + low) * imageSize + low;
        result[labels[labelY + x]] = 1.0f;
        for (int i = 0; i < patchSize; i++) {
            for (int j = 0; j < patchSize; j++) {
                int imagePos = ((y + i) * imageSize + x + j) * 3;
                int patchPos = (i * patchSize + j) * 3;
                result[patchPos + classesCount] = image[imagePos];
                result[patchPos + classesCount + 1] = image[imagePos + 1];
                result[patchPos + classesCount + 2] = image[imagePos + 2];
            }
        }
        return result;
    }
}
