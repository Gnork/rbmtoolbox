package berlin.iconn.projects.segmentation.multiscale;

import berlin.iconn.projects.segmentation.SegmentationDataConverter;
import berlin.iconn.rbm.IRBM;
import berlin.iconn.rbm.enhancements.RBMInfoPackage;
import berlin.iconn.rbm.enhancements.visualizations.IVisualizeObserver;
import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;
import org.jblas.ranges.IntervalRange;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.Arrays;

/**
 * Created by Moritz on 7/15/2014.
 */
public class ShowMultiScaleLabels extends JComponent implements IVisualizeObserver {

    private final MultiImageWithLabels data;
    private final MultiScaleProvider provider;
    private final int classesCount;
    private final IRBM[] rbms;
    private final int patchSize;
    private final int edgeLength;
    private final int patchSizeHalf;
    private RBMInfoPackage info;
    private int state = 1;

    private final BufferedImage[] originalClasses;
    private final BufferedImage[] reconstructedClasses;
    private final float[][] labels;
    private final Dimension dim;

    public ShowMultiScaleLabels(MultiImageWithLabels data, MultiScaleProvider provider, int classesCount, IRBM[] rbms) {
        this.data = data;
        this.provider = provider;
        this.classesCount = classesCount;
        this.rbms = rbms;

        this.originalClasses = new BufferedImage[classesCount];
        this.reconstructedClasses = new BufferedImage[classesCount];

        dim = new Dimension(classesCount * 64, 128);

        int[] imageSizes = provider.getImageSizes();
        this.patchSize = imageSizes[imageSizes.length - 1];
        this.patchSizeHalf = patchSize / 2;
        this.edgeLength = imageSizes[imageSizes.length - 2] - patchSize + 1;

        this.labels = applyLabels();

        for (int i = 0; i < originalClasses.length; i++) {
            originalClasses[i] = new BufferedImage(edgeLength, edgeLength, BufferedImage.TYPE_INT_RGB);
            reconstructedClasses[i] = new BufferedImage(edgeLength, edgeLength, BufferedImage.TYPE_INT_RGB);
        }

        for (int i = 0; i < edgeLength; i++) {
            for (int j = 0; j < edgeLength; j++) {
                classesToImage(originalClasses, labels[i * edgeLength + j], j, i);
            }
        }
    }


    @Override
    public Dimension getPreferredSize() {
        return dim;
    }

//    protected final float sumLabel(int[] reconstructionLabels) {
//        float sum = 0.0f;
//        for (int i = 0; i < reconstructionLabels.length; i++) {
//                sum += reconstructionLabels[i] == data.labels[i] ? 0.0f : 1.0f;
//        }
//        return  sum;
//    }

    private final float sumSquare(float[] original, float[] reconstruction) {
        float sum = 0.0f;
        float diff;
        for (int i = 0; i < original.length; i++) {
            diff = original[i] - reconstruction[i];
            sum += diff * diff;
        }
        return sum;
    }

    private void classesToImage(BufferedImage[] images, float[] labels, int x, int y) {

        for (int i = 0; i < classesCount; i++) {
            int value = (int)(labels[i] * 255.0f);
            images[i].setRGB(x, y, (value << 16) | (value << 8) | value);
        }
    }



    private float[] process() {
        float[] errors = new float[2];
        float mseSum  = 0.0f;
        float labelSum = 0.0f;

        for (int i = 0; i < edgeLength; i++) {
            for (int j = 0; j < edgeLength; j++) {
                float[] feature = provider.getFeatures(data, j, i, false);
                float[] recon = reconstruction(feature);
                final float[] label = labels[i * edgeLength + j];
                classesToImage(reconstructedClasses, recon, j, i);
                mseSum += sumSquare(label, recon);
                if(info != null) {
                    if(info.getEpochs() == 800) {
                        float bla = 0.0f;
                    }
                }
            }
        }
        int length = edgeLength * edgeLength;
        errors[0] = labelSum / length;
        errors[1] = (float) Math.sqrt(mseSum / (length * classesCount));
        System.out.println(errors[1]);
        return errors;
    }

    private float[] reconstruction(float[] patch) {
        float[] recon1 = Arrays.copyOf(patch, patch.length);
        float[][] recon = new float[][]{ recon1 };
        for (int i = 0; i < state; i++) {
            recon = rbms[i].getHidden(recon);
        }
        for (int i = state - 1; i >= 0; i--) {
            recon = rbms[i].getVisible(recon);
        }
        return recon[0];
    }

    public void nextState() {
        if(++state >= rbms.length) {
            state = rbms.length;
        }
    }
    private int getState() {
        return state;
    }

    @Override
    public void update(RBMInfoPackage pack) {
        this.info = pack;
        process();

        paintImmediately(0, 0, dim.width, dim.height);
    }

    public float[][] applyLabels() {
        float[][] result = new float[edgeLength * edgeLength][classesCount];
        for (int i = 0; i < edgeLength; i++) {
            final int y = edgeLength * i;
            final int yL = (edgeLength + patchSize - 1) * (patchSizeHalf + i);
            for (int j = 0; j < edgeLength; j++) {
                final int pos = y + j;
                final int posL = yL + j + patchSizeHalf;
                result[pos][data.labels[posL]] = 1.0f;
            }
        }
        return result;
    }

    @Override
    public void paint(Graphics graphics) {
        for (int i = 0; i < originalClasses.length; i++) {
//            graphics.drawImage(img, dx1, dy1, dx2, dy2, sx1, sy1, sx2, sy2,null);
            graphics.drawImage(originalClasses[i], i * 64, 0, i * 64 + 64, 64, 0, 0, edgeLength, edgeLength, null);
            graphics.drawImage(reconstructedClasses[i], i * 64, 64, i * 64 + 64, 128, 0, 0, edgeLength, edgeLength, null);
        }

        if(info != null) {


        }
    }
}
