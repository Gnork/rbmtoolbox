/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package berlin.iconn.rbm.binarize;

import java.util.Random;
import org.jblas.FloatMatrix;

/**
 *
 * @author christoph
 */
public class CommonBinarize implements IBinarize{
    Random random = new Random();

    @Override
    public FloatMatrix binarize(FloatMatrix data) {
        float[][] randomMatrix = FloatMatrix.rand(data.getRows(), data.getColumns()).toArray2();

        float[][] tmpHiddenStates = data.dup().toArray2();
        for (int y = 0; y < tmpHiddenStates.length; y++) {
            for (int x = 0; x < tmpHiddenStates[y].length; x++) {
                tmpHiddenStates[y][x] = (tmpHiddenStates[y][x] > randomMatrix[y][x]) ? 1 : 0;
            }
        }
        return new FloatMatrix(tmpHiddenStates);
    }
}
