package berlin.iconn.projects.segmentation.smallsegmentation;

import berlin.iconn.projects.segmentation.SegmentationDataConverter;
import berlin.iconn.rbm.IRBM;
import org.jblas.FloatMatrix;
import org.jblas.ranges.IntervalRange;

/**
 * Created by Moritz on 6/4/2014.
 */
public class ShowSegmentation1 extends AShowSegmentation {

    private final IRBM rbmLabels;
    private final IRBM rbmImages;
    private final IRBM rbmCombination;
    private final IRBM rbmAssociation;

    private int state = 0;


    public ShowSegmentation1(int[] labelimage, float[] image, int width, int height,
                             IRBM rbmLabels,
                             IRBM rbmImages,
                             IRBM rbmCombination,
                             IRBM rbmAssociation,
                             int classLength,
                             int patchSize) {
        super(labelimage, image, patchSize, classLength, width, height);

        this.rbmLabels = rbmLabels;
        this.rbmImages = rbmImages;
        this.rbmCombination = rbmCombination;
        this.rbmAssociation = rbmAssociation;

    }

    public void nextState() {
        state++;
    }


    @Override
    protected void process() {
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

}
