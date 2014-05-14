package berlin.iconn.scanpicture;

import berlin.iconn.rbm.RBM;
import berlin.iconn.rbm.StoppingCondition;
import berlin.iconn.rbm.WeightsFactory;
import berlin.iconn.rbm.dataprovider.FilterPictureBatchProvider;
import berlin.iconn.rbm.dataprovider.RandomPictureBatchSelectionProvider;
import berlin.iconn.rbm.enhancements.RBMEnhancer;
import berlin.iconn.rbm.enhancements.TrainingVisualizer;
import berlin.iconn.rbm.learningRate.ConstantLearningRate;
import berlin.iconn.rbm.GrowingModifier;
import org.apache.commons.io.FileUtils;
import org.jblas.FloatMatrix;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;

/**
 * Created by Moritz on 4/28/2014.
 */
public class Main {

    private static final boolean exportImages = true;
    private static final String exportPath = "export";
    private static final int edgeLength = 256;
    private static final int padding = 0;
    private static final boolean isRGB = false;
    private static final boolean binarize = false;
    private static final boolean invert = false;
    private static final float minData = 0.0f;
    private static final float maxData = 1.0f;
    private static final String picture = "Data/Pictures";
    public static void main(String[] args) {

        int rbmEdgeLength = 8;
        DataSet[] trainingDataSet = loadData(picture);
        final float[][] trainingData = dataSetToArray(trainingDataSet);

        RBMEnhancer enhancer = new RBMEnhancer(new RBM(WeightsFactory.randomGaussianWeightsWithBias(rbmEdgeLength * rbmEdgeLength, rbmEdgeLength / 2, 0.01f), new GrowingModifier()));

        ScanPicture picture = new ScanPicture(new FloatMatrix(edgeLength, edgeLength, trainingData[0]), rbmEdgeLength);
        new Frame(picture);

        enhancer.addEnhancement(new TrainingVisualizer(1,picture));

        FloatMatrix[] batchSelectionData =  new FloatMatrix[trainingData.length];
        //prepare data for batch selection
        for (int i = 0; i < trainingData.length; i++) {
            batchSelectionData[i] = new FloatMatrix(edgeLength, edgeLength, trainingData[i]);
        }

        enhancer.train(new RandomPictureBatchSelectionProvider( batchSelectionData, 2, rbmEdgeLength, rbmEdgeLength ),
                new StoppingCondition(1000000),
                new ConstantLearningRate(0.2f));
    }
    /**
     * loads the image data from a directory and converts into a data structure
     *
     * @param importPath
     * @return
     */
    public static DataSet[] loadData(String importPath) {

        File imageFolder = new File(importPath);
        final File[] imageFiles = imageFolder.listFiles(new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                return (name.endsWith("jpg") || name.endsWith("png") || name.endsWith("gif"));
            }
        });

        int size = edgeLength * edgeLength;
        DataSet[] result = new DataSet[imageFiles.length];

        for (int i = 0; i < imageFiles.length; i++) {
            float[] imageData;
            try {
                imageData = DataConverter.processPixelData(ImageIO.read(imageFiles[i]), edgeLength, binarize, invert, minData, maxData, isRGB);
            } catch (IOException e) {
                System.out.println("Could not load: " + imageFiles[i].getAbsolutePath());
                return null;
            }

            imageData = pad(imageData, edgeLength, padding);

            String label = imageFiles[i].getName().split("_")[0];
            result[i] = new DataSet(imageData, label);

        }

        return result;
    }

    public static float[][] dataSetToArray(DataSet[] dataSet) {
        float[][] result = new float[dataSet.length][];
        for (int i = 0; i < dataSet.length; ++i) {
            result[i] = dataSet[i].getData();
        }
        return result;
    }

    public static void deleteOldExportData() {
        try {
            FileUtils.deleteDirectory(new File(exportPath));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public static void exportAsImage(float[][] data, String name, int count) {
        for (int i = 0; i < data.length; i++) {
            exportAsImage(data[i], name, count, i);
        }
    }

    public static void exportAsImage(float[] data, String name, int count, int index) {
        if (!exportImages) return;
        new File(exportPath + "/" + name + "/").mkdirs();

        BufferedImage image = DataConverter.pixelDataToImage(data, 0.0f, false);
        File outputfile = new File(exportPath + "/" + name + "/" + count + "_" + index + ".png");
        try {
            ImageIO.write(image, "png", outputfile);
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    /**
     * adds a padding to the training images, so that the filtered image
     * has the original size
     *
     * @param data
     * @param dataEdgeLength
     * @param padding
     * @return
     */
    private static float[] pad(float[] data, int dataEdgeLength, int padding) {
        int newEdgeLength = dataEdgeLength + padding * 2;
        float[] result = new float[newEdgeLength * newEdgeLength];

        for (int y = 0; y < newEdgeLength; y++) {
            for (int x = 0; x < newEdgeLength; x++) {

                int pos = y * newEdgeLength + x;
                if (y < padding || x < padding || y >= dataEdgeLength + padding || x >= dataEdgeLength + padding) {
                    result[pos] = 0.0f;
                } else {
                    int posm = (y - padding) * (newEdgeLength - padding * 2) + x - padding;
                    result[pos] = data[posm];
                }

            }
        }

        return result;
    }
}
