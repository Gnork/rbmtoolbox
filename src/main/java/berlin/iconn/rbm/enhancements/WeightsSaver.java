/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package berlin.iconn.rbm.enhancements;

import berlin.iconn.persistence.Persistor;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author christoph
 */
public class WeightsSaver implements IRBMTrainingEnhancement{
    
    private final int updateInterval;
    private final Persistor persistor;
    
    public WeightsSaver(int updateInterval) {
        super();
        this.updateInterval = updateInterval;
        this.persistor = new Persistor();
    }

    @Override
    public int getUpdateInterval() {
        return updateInterval;
    }

    @Override
    public void action(RBMInfoPackage info) {
        try {
            persistor.overwriteSimpleWeights(info.getWeights());
            System.out.println("Saved Weights");
        } catch (IOException ex) {
            Logger.getLogger(WeightsSaver.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
}
