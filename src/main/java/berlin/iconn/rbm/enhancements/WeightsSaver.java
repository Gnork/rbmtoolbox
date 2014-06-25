/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package berlin.iconn.rbm.enhancements;

import berlin.iconn.persistence.InOutOperations;
import java.io.IOException;
import java.util.Date;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author christoph
 */
public class WeightsSaver implements IRBMTrainingEnhancement{
    
    private final int updateInterval;
    private final Date date;
    private final String suffix;
    
    public WeightsSaver(int updateInterval) {      
        this(new Date(), updateInterval, "weights");
    }
    
    public WeightsSaver(Date date, int updateInterval, String suffix){
        super();
        this.date = date;
        this.updateInterval = updateInterval;
        this.suffix = suffix;
    }

    @Override
    public int getUpdateInterval() {
        return updateInterval;
    }

    @Override
    public void action(RBMInfoPackage info) {
        try {
            InOutOperations.saveSimpleWeights(info.getWeights(), date, suffix);
            System.out.println("Weights saved!");
        } catch (IOException ex) {
            Logger.getLogger(WeightsSaver.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
}
