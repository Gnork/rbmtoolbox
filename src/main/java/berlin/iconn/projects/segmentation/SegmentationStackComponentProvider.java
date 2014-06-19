/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package berlin.iconn.projects.segmentation;

import berlin.iconn.rbm.dataprovider.ATrainingDataProvider;
import org.jblas.FloatMatrix;

/**
 *
 * @author christoph
 */
public class SegmentationStackComponentProvider extends ATrainingDataProvider{

    public SegmentationStackComponentProvider(FloatMatrix data) {
        super(data);
    }
    
    public void setDataForTraining(FloatMatrix data){
        super.setData(data);
    }

    @Override
    public FloatMatrix getDataWithBiasForTraining() {
        return super.getDataWithBias();
    }

    @Override
    public FloatMatrix getTransposedDataWithBiasForTraining() {
        return super.getTransposedDataWithBias();
    }

    @Override
    public void changeDataAtTraining() {

    }

    
}
