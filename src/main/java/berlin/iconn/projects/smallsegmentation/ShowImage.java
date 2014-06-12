package berlin.iconn.projects.smallsegmentation;

import berlin.iconn.rbm.IRBM;
import berlin.iconn.rbm.enhancements.visualizations.IVisualizeObserver;
import org.jblas.FloatMatrix;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.HashMap;
import java.util.Random;

/**
 * Created by Moritz on 6/4/2014.
 */
public class ShowImage extends JComponent {

    private static final HashMap<Integer, Integer> colorLabelMap = generateTable();
    private final FloatMatrix labelMatrix;
    private final FloatMatrix imagePatchMatrix;
    private final int width, height;
    private final Dimension dim = new Dimension(800, 600);
    private final IRBM rbmLabels;
    private final IRBM rbmImages;
    private final IRBM rbmCombination;
    private final IRBM rbmAssociation;
    private final BufferedImage bufferedImage;
    private final BufferedImage bufferedLabels;
    private final int classLength;


    public ShowImage(int[] labelimage, float[] image, int width, int height,
                     IRBM rbmLabels,
                     IRBM rbmImages,
                     IRBM rbmCombination,
                     IRBM rbmAssociation,
                     int classLength) {

        this.width = width;
        this.height = height;
        this.rbmLabels = rbmLabels;
        this.rbmImages = rbmImages;
        this.rbmCombination = rbmCombination;
        this.rbmAssociation = rbmAssociation;

        this.classLength = classLength;
        FloatMatrix[] data = SegmentationDataConverter.createTrainingData(labelimage, image, width, height, 5, classLength);

        this.labelMatrix = data[0];
        this.imagePatchMatrix = data[1];

        this.bufferedLabels = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

        this.bufferedImage = SegmentationDataConverter.getImageData(imagePatchMatrix.toArray2(), width, height, 5);


        final int[] labelData = SegmentationDataConverter.getLabelData(labelMatrix.toArray2(), width, height, 5);
        final int[] pixelsOfLabels = getPixelsOfLabels(labelData);
        bufferedLabels.setRGB(0,0, width,height, pixelsOfLabels, 0, width);
    }

    private static HashMap<Integer, Integer> generateTable() {
        HashMap<Integer, Integer> result = new HashMap<>();
        Random random = new Random();
        for (int i = 0; i < 128; i++) {
            int r = random.nextInt(16);
            int g = random.nextInt(16);
            int b = random.nextInt(16);
            r |= r << 4;
            g |= g << 4;
            b |= b << 4;
            result.put(i, (r << 16) | (g << 8) | b);
        }
        return result;
    }

    @Override
    public Dimension getPreferredSize() {
        return dim;
    }


    private int[] getPixelsOfLabels(int[] labels) {
        int[] result = new int[width * height];

        for (int i = 0; i < height; i++) {
            final int YResult = i * width;
            for (int j = 0; j < width; j++) {
                result[YResult + j] = colorLabelMap.get(labels[YResult + j]);
            }
        }
        return result;
    }

    private int[] getPixelsOfImage(float[] image) {
        int[] result = new int[width * height];

        for (int i = 0; i < height; i++) {

            final int y = i * width * 3;
            for (int j = 0; j < width; j++) {
                int index = j * 3;
                final int position = y + index;
                int r = (int) (image[position] * 255) << 16;
                int g = (int) (image[position + 1] * 255) << 8;
                int b = (int) (image[position + 2] * 255);
                result[i * width + j] = r | g | b;
            }
        }
        return result;
    }

    @Override
    public void paint(Graphics graphics) {

            graphics.drawImage(bufferedImage,0,0,null);
            graphics.drawImage(bufferedLabels, width,0, null);

    }

}
