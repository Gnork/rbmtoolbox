package berlin.iconn.projects.smallsegmentation;

import berlin.iconn.rbm.IRBM;
import berlin.iconn.rbm.enhancements.RBMInfoPackage;
import berlin.iconn.rbm.enhancements.visualizations.IVisualizeObserver;
import org.jblas.FloatMatrix;
import org.jblas.ranges.IntervalRange;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.HashMap;
import java.util.Random;

/**
 * Created by Moritz on 6/4/2014.
 */
public class ShowSegmentation extends JComponent implements IVisualizeObserver {

    private static final HashMap<Integer, Integer> colorLabelMap = generateTable();
    private final FloatMatrix labelMatrix;
    private final FloatMatrix imagePatchMatrix;
    private final int width, height;
    private final Dimension dim = new Dimension(400, 400);
    private final IRBM rbmLabels;
    private final IRBM rbmImages;
    private final IRBM rbmCombination;
    private final IRBM rbmAssociation;
    private final BufferedImage bufferedImage;
    private final BufferedImage bufferedLabels;

    private final BufferedImage resultImage;
    private final BufferedImage resultLabels;
    private final int classLength;
    private final int patchSize;

    private RBMInfoPackage info = null;

    private int state = 0;


    public ShowSegmentation(int[] labelimage, float[] image, int width, int height,
                            IRBM rbmLabels,
                            IRBM rbmImages,
                            IRBM rbmCombination,
                            IRBM rbmAssociation,
                            int classLength,
                            int patchSize) {

        this.width = width;
        this.height = height;
        this.rbmLabels = rbmLabels;
        this.rbmImages = rbmImages;
        this.rbmCombination = rbmCombination;
        this.rbmAssociation = rbmAssociation;

        this.patchSize = patchSize;
        this.classLength = classLength;
        FloatMatrix[] data = SegmentationDataConverter.createTrainingData(labelimage, image, width, height, patchSize, classLength);

        this.labelMatrix = data[0];
        this.imagePatchMatrix = data[1];

        this.resultLabels = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        this.resultImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

        this.bufferedLabels = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        this.bufferedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
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

    public void nextState() {
        state++;
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

    private void process() {
        int[] pixels = null;
        int[] labels = null;
        float[][] hiddenImagePatches = rbmImages.getHidden(imagePatchMatrix.toArray2());
        if(state == 0) {
            float[][] hiddenLabels = rbmLabels.getHidden(labelMatrix.toArray2());
            float[][] visibleLabels = rbmLabels.getVisible(hiddenLabels);
            labels = SegmentationDataConverter.getLabelData(visibleLabels, width, height, patchSize);
            float[][] visiblePatches = rbmImages.getVisible(hiddenImagePatches);
            pixels = SegmentationDataConverter.getImageData(visiblePatches, width, height, patchSize);
        }
        if(state == 1) {

            float[][] combination = FloatMatrix.concatHorizontally(
                    FloatMatrix.zeros(labelMatrix.getRows(),rbmLabels.getWeights()[0].length - 1),
                    new FloatMatrix(hiddenImagePatches)).toArray2();
            float[][] hiddenCombinaton = rbmCombination.getHidden(combination);
            float[][] visibleCombination = rbmCombination.getVisible(hiddenCombinaton);

            FloatMatrix combinationVisibleMatrix = new FloatMatrix(visibleCombination);
            float[][] hiddenLabels =  combinationVisibleMatrix.get(new IntervalRange(0, labelMatrix.getRows()),
                    new IntervalRange(0, labelMatrix.getColumns())).toArray2();
            float[][] hiddenPatches = combinationVisibleMatrix.get(new IntervalRange(0, labelMatrix.getRows()),
                    new IntervalRange(labelMatrix.getColumns(), combinationVisibleMatrix.getColumns())).toArray2();

            float[][] visibleLabels = rbmLabels.getVisible(hiddenLabels);
            labels = SegmentationDataConverter.getLabelData(visibleLabels, width, height, patchSize);
            float[][] visiblePatches = rbmImages.getVisible(hiddenPatches);
            pixels = SegmentationDataConverter.getImageData(visiblePatches, width, height, patchSize);
        }
        if(state > 1) {

            float[][] combination = FloatMatrix.concatHorizontally(
                    FloatMatrix.zeros(labelMatrix.getRows(),rbmLabels.getWeights()[0].length - 1),
                    new FloatMatrix(hiddenImagePatches)).toArray2();
            float[][] hiddenCombinaton = rbmCombination.getHidden(combination);
            float[][] hiddenAssociation = rbmAssociation.getHidden(hiddenCombinaton);
            float[][] visiblenAssociation = rbmAssociation.getVisible(hiddenAssociation);
            float[][] visibleCombination = rbmCombination.getVisible(visiblenAssociation);

            FloatMatrix combinationVisibleMatrix = new FloatMatrix(visibleCombination);
            float[][] hiddenLabels =  combinationVisibleMatrix.get(new IntervalRange(0, labelMatrix.getRows()),
                    new IntervalRange(0, labelMatrix.getColumns())).toArray2();
            float[][] hiddenPatches = combinationVisibleMatrix.get(new IntervalRange(0, labelMatrix.getRows()),
                    new IntervalRange(labelMatrix.getColumns(), combinationVisibleMatrix.getColumns())).toArray2();

            float[][] visibleLabels = rbmLabels.getVisible(hiddenLabels);
            labels = SegmentationDataConverter.getLabelData(visibleLabels, width, height, patchSize);
            float[][] visiblePatches = rbmImages.getVisible(hiddenPatches);
            pixels = SegmentationDataConverter.getImageData(visiblePatches, width, height, patchSize);
        }


        resultImage.setRGB(0,0,width,height, pixels, 0, width);
        resultLabels.setRGB(0,0,width,height, getPixelsOfLabels(labels), 0, width);
    }

    @Override
    public void paint(Graphics graphics) {

        graphics.drawImage(bufferedImage,0,0,null);
        graphics.drawImage(bufferedLabels, width,0, null);

        process();

        graphics.drawImage(resultImage, 0, height, null);
        graphics.drawImage(resultLabels, width, height, null);

        if(info != null) {
            graphics.setColor(Color.black);
            graphics.drawString("Epochs: " + info.getEpochs(), 40, 320);
        }
    }

    @Override
    public void update(RBMInfoPackage pack) {
        this.info = pack;
        paintImmediately(0,0,dim.width, dim.height);
    }
}
