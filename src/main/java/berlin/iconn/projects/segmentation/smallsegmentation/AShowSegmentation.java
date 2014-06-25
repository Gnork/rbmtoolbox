package berlin.iconn.projects.segmentation.smallsegmentation;

import berlin.iconn.projects.segmentation.SegmentationDataConverter;
import berlin.iconn.rbm.enhancements.RBMInfoPackage;
import berlin.iconn.rbm.enhancements.visualizations.IVisualizeObserver;
import org.jblas.FloatMatrix;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.HashMap;
import java.util.Random;

/**
 * Created by Moritz on 6/24/2014.
 */
public abstract class AShowSegmentation extends JComponent implements IVisualizeObserver {
    private static final HashMap<Integer, Integer> colorLabelMap = generateTable();
    protected final FloatMatrix labelMatrix;
    protected final FloatMatrix imagePatchMatrix;
    protected final int width;
    protected final int height;
    protected final BufferedImage bufferedImage;
    protected final BufferedImage bufferedLabels;
    protected final BufferedImage resultImage;
    protected final BufferedImage resultLabels;
    protected final int classLength;
    protected final int patchSize;
    private final Dimension dim;
    private RBMInfoPackage info = null;

    public AShowSegmentation(int[] labelimage, float[] image, int patchSize, int classLength, int width, int height) {
        this.patchSize = patchSize;
        this.classLength = classLength;
        this.bufferedLabels = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        this.resultImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        this.bufferedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        this.resultLabels = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        this.width = width;
        this.height = height;
        dim = new Dimension(width * 2, height * 2 + 100);

        FloatMatrix[] data = SegmentationDataConverter.createTrainingData(labelimage, image, width, height, patchSize, classLength);

        this.labelMatrix = data[0];
        this.imagePatchMatrix = data[1];
        ;

        this.bufferedImage.setRGB(0, 0, width, height,
                SegmentationDataConverter.getImageData(imagePatchMatrix.toArray2(), width, height, patchSize),
                0, width);

        final int[] labelData = SegmentationDataConverter.getLabelData(labelMatrix.toArray2(), width, height, patchSize);
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

    protected int[] getPixelsOfLabels(int[] labels) {
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

    protected abstract void process();

    @Override
    public void paint(Graphics graphics) {

        graphics.drawImage(bufferedImage,0,0,null);
        graphics.drawImage(bufferedLabels, width,0, null);

        process();

        graphics.drawImage(resultImage, 0, height, null);
        graphics.drawImage(resultLabels, width, height, null);

        if(info != null) {
            graphics.setColor(Color.black);
            graphics.drawString("Epochs: " + info.getEpochs(), 20, height * 2 + 40);
        }
    }

    @Override
    public void update(RBMInfoPackage pack) {
        this.info = pack;
        paintImmediately(0,0,dim.width, dim.height);
    }
}
