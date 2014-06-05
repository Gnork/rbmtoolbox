package berlin.iconn.projects.smallsegmentation;

import org.jblas.FloatMatrix;

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
                float[] patch = new float[75];
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
}
