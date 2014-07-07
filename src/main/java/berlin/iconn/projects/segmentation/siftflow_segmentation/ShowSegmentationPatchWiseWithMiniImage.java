package berlin.iconn.projects.segmentation.siftflow_segmentation;

import berlin.iconn.projects.segmentation.SegmentationDataConverter;
import berlin.iconn.projects.segmentation.smallsegmentation.AShowSegmentation;
import berlin.iconn.rbm.IRBM;
import org.jblas.FloatMatrix;
import org.jblas.ranges.IntervalRange;

/**
 * Created by Moritz on 6/24/2014.
 */
public class ShowSegmentationPatchWiseWithMiniImage extends AShowSegmentation {

    private final IRBM combinationRBM;
    private final FloatMatrix miniImage;
    private float error = 1.0f;
    private int state = 0;
    public ShowSegmentationPatchWiseWithMiniImage(int[] labelimage, float[] image, float[] miniImage, int patchSize, int classLength, int width, int height, IRBM combinationRBM) {
        super(labelimage, image, patchSize, classLength, width, height);
        this.miniImage = FloatMatrix.zeros(1, miniImage.length);
        this.miniImage.data = miniImage;
        this.combinationRBM = combinationRBM;
    }

    public void nextState() {
        state++;
    }

    @Override
    protected float[] process(FloatMatrix labelMatrix) {

        float[] errors = new float[2];
        FloatMatrix combinationMatrix = FloatMatrix.concatHorizontally(
                FloatMatrix.zeros(labelMatrix.getRows(), labelMatrix.getColumns()), imagePatchMatrix);
        FloatMatrix miniImageMatrix = miniImage.repmat(combinationMatrix.rows, 1);
        combinationMatrix = FloatMatrix.concatHorizontally(combinationMatrix, miniImageMatrix);

        float[][] combination = combinationMatrix.toArray2();
        float[][] hidden = combinationRBM.getHidden(combination);



        FloatMatrix combinationVisibleMatrix = new FloatMatrix(combinationRBM.getVisible(hidden));
        FloatMatrix visibleLabels =  combinationVisibleMatrix.get(new IntervalRange(0, labelMatrix.getRows()),
                new IntervalRange(0, labelMatrix.getColumns()));

        errors[0] = getMSE(visibleLabels);
        float[][] visiblePatches = combinationVisibleMatrix.get(new IntervalRange(0, labelMatrix.getRows()),
                new IntervalRange(labelMatrix.getColumns(), combinationVisibleMatrix.getColumns())).toArray2();

        int[] labels = SegmentationDataConverter.getLabelData(visibleLabels.toArray2(), width, height, patchSize);
        errors[1] = getLabelError(labels);
        int[] pixels = SegmentationDataConverter.getImageData(visiblePatches, width, height, patchSize);

        resultImage.setRGB(0,0,width,height, pixels, 0, width);
        resultLabels.setRGB(0,0,width,height, getPixelsOfLabels(labels), 0, width);
        return errors;
    }
}
