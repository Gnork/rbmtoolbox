/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package berlin.iconn.rbm.enhancements;

/**
 *
 * @author christoph
 */
public class InfoLogger implements IRBMTrainingEnhancement{
    
    private final int updateInterval;
    
    public InfoLogger(int updateInterval){
        this.updateInterval = updateInterval;
    }

    @Override
    public int getUpdateInterval() {
        return updateInterval;
    }

    @Override
    public void action(RBMInfoPackage info) {
        System.out.println("epoch: " + info.getEpochs() + ", error: " + info.getError());
    }
    
}
