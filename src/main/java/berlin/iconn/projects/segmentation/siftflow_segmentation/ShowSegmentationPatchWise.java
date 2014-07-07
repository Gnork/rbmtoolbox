package berlin.iconn.projects.segmentation.siftflow_segmentation;

import berlin.iconn.projects.segmentation.SegmentationDataConverter;
import berlin.iconn.projects.segmentation.smallsegmentation.AShowSegmentation;
import berlin.iconn.rbm.IRBM;
import org.jblas.FloatMatrix;
import org.jblas.ranges.IntervalRange;

/**
 * Created by Moritz on 6/24/2014.
 */
public class ShowSegmentationPatchWise extends AShowSegmentation {

    private final IRBM combinationRBM;
    private float error = 1.0f;
    private int state = 0;
    public ShowSegmentationPatchWise(int[] labelimage, float[] image, int patchSize, int classLength, int width, int height, IRBM combinationRBM) {
        super(labelimage, image, patchSize, classLength, width, height);
        this.combinationRBM = combinationRBM;
    }

    @Override
    protected float[] process(FloatMatrix labelMatrix) {

        float[] errors = new float[2];
        float[][] combination = FloatMatrix.concatHorizontally(
                FloatMatrix.zeros(labelMatrix.getRows(), labelMatrix.getColumns()), imagePatchMatrix).toArray2();
        float[][] hidden = combinationRBM.getHidden(combination);


        FloatMatrix combinationVisibleMatrix = new FloatMatrix(combinationRBM.getVisible(hidden));
        final FloatMatrix visibleLabelMatrix = combinationVisibleMatrix.get(new IntervalRange(0, labelMatrix.getRows()),
                new IntervalRange(0, labelMatrix.getColumns()));
        errors[0] = getMSE(visibleLabelMatrix);

        float[][] visibleLabels =  visibleLabelMatrix.toArray2();
        float[][] visiblePatches = combinationVisibleMatrix.get(new IntervalRange(0, labelMatrix.getRows()),
                new IntervalRange(labelMatrix.getColumns(), combinationVisibleMatrix.getColumns())).toArray2();

        int[] labels = SegmentationDataConverter.getLabelData(visibleLabels, width, height, patchSize);
        errors[1] = getLabelError(labels);
        int[] pixels = SegmentationDataConverter.getImageData(visiblePatches, width, height, patchSize);

        resultImage.setRGB(0,0,width,height, pixels, 0, width);
        resultLabels.setRGB(0,0,width,height, getPixelsOfLabels(labels), 0, width);
        return errors;
    }
}
