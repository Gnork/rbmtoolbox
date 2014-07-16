package berlin.iconn.projects.segmentation.smallsegmentation;

import berlin.iconn.projects.segmentation.SegmentationDataConverter;
import berlin.iconn.rbm.IRBM;
import org.jblas.FloatMatrix;
import org.jblas.ranges.IntervalRange;

import java.awt.*;
import java.awt.image.BufferedImage;

/**
 * Created by Moritz on 6/24/2014.
 */
public class ShowSegmentationClasses extends AShowSegmentation {

    private final IRBM combinationRBM;
    private final IRBM associationRBM;
    private final BufferedImage[] originalClasses;
    private final BufferedImage[] reconstructedClasses;
    private int state = 0;
    public ShowSegmentationClasses(int[] labelimage, float[] image, int patchSize, int classLength, int width, int height, IRBM combinationRBM, IRBM associationRBM) {
        super(labelimage, image, patchSize, classLength, width, height);
        this.combinationRBM = combinationRBM;
        this.associationRBM = associationRBM;
        dim = new Dimension(classLength * 64, 128);

        this.originalClasses = new BufferedImage[classLength];
        classestoImage(originalClasses, labelMatrix);
        this.reconstructedClasses = new BufferedImage[classLength];
    }



    private void classestoImage(BufferedImage[] classImages, FloatMatrix labelMatrix) {
        int newWidth = width - patchSize + 1;
        int newHeight = height - patchSize + 1;
        float[][] labels = labelMatrix.transpose().toArray2();
        for (int i = 0; i < classImages.length; i++) {
            int[] pixels = new int[newWidth * newHeight];
            for (int j = 0; j < newHeight; j++) {
                for (int k = 0; k < newWidth; k++) {
                    int pos = j * newWidth + k;
                    int value = (int) (labels[i][pos] * 255f);
                    pixels[pos] = 0xFF << 24 | value << 16 | value << 8 | value;
                }
            }
            classImages[i] = new BufferedImage(newWidth, newHeight, BufferedImage.TYPE_INT_RGB);
            classImages[i].setRGB(0, 0, newWidth, newHeight, pixels, 0, newWidth);
        }
    }


    public void nextState() {
        state++;
    }

    @Override
    protected float[] process(FloatMatrix labelMatrix) {
        float[] errors = new float[2];
        float[][] combination = FloatMatrix.concatHorizontally(
                labelMatrix,
                imagePatchMatrix).toArray2();


        float[][] hidden = combinationRBM.getHidden(combination);

        if (state > 0) {
          hidden = associationRBM.getVisible(associationRBM.getHidden(hidden));
        }

        FloatMatrix combinationVisibleMatrix = new FloatMatrix(combinationRBM.getVisible(hidden));

        final FloatMatrix visibleLabelMatrix = combinationVisibleMatrix.get(new IntervalRange(0, labelMatrix.getRows()),
                new IntervalRange(0, labelMatrix.getColumns()));
        errors[0] = getMSE(visibleLabelMatrix);
        classestoImage(reconstructedClasses, visibleLabelMatrix);
        float[][] visibleLabels =  visibleLabelMatrix.toArray2();

        int[] labels = SegmentationDataConverter.getLabelData(visibleLabels, width, height, patchSize);
        errors[1] = getLabelError(labels);

        return errors;
    }

    @Override
    public void paint(Graphics graphics) {

//        float[] errors =  process(labelMatrix);
//        mse = errors[0];
//        labelError = errors[1];

        float[] errorsZero =  process(FloatMatrix.rand(labelMatrix.getRows(), labelMatrix.getColumns()).sub(0.5f).mul(0.02f));

        for (int i = 0; i < originalClasses.length; i++) {
//            graphics.drawImage(img, dx1, dy1, dx2, dy2, sx1, sy1, sx2, sy2,null);
            graphics.drawImage(originalClasses[i], i * 64, 0, i * 64 + 64, 64, 0, 0, width, height, null);
            graphics.drawImage(reconstructedClasses[i], i * 64, 64, i * 64 + 64, 128, 0, 0, width, height, null);
        }

        if(info != null) {


        }
    }


}
