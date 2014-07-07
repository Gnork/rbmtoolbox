/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package berlin.iconn.projects.facerepair;

import org.jblas.FloatMatrix;

/**
 *
 * @author christoph
 */
public class rangeMatrixTest {
    public static void main(String[] args){
        float[][] test = new float[][]{{3,4},{5,6}};
        int i = 0;
        FloatMatrix testMatrix = new FloatMatrix(test);
        i++;
        FloatMatrix columnAdded = FloatMatrix.concatHorizontally(FloatMatrix.ones(testMatrix.getRows(), 1), testMatrix);
        i++;
        FloatMatrix columnRemoved = columnAdded.getRange(0, columnAdded.getRows(), 1, columnAdded.getColumns());
        i++;
    }
}
