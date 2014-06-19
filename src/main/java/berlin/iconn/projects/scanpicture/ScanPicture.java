package berlin.iconn.projects.scanpicture;

import berlin.iconn.rbm.RBM;
import berlin.iconn.rbm.enhancements.RBMInfoPackage;
import berlin.iconn.rbm.enhancements.visualizations.IVisualizeObserver;
import org.jblas.FloatMatrix;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;

/**
 * Created by Moritz on 4/29/2014.
 */
public class ScanPicture extends JComponent implements IVisualizeObserver {

    RBMInfoPackage info = null;
    final int rbmEdgeLength;
    final int pictureWidth;
    final int pictureHeight;
    final FloatMatrix originalData;
    final BufferedImage originalImage;

    public ScanPicture(FloatMatrix data, int rbmEdgeLength) {
        this.rbmEdgeLength = rbmEdgeLength;
        this.pictureWidth = rbmEdgeLength * (int) Math.ceil(data.getColumns() / (double) rbmEdgeLength);
        this.pictureHeight = rbmEdgeLength * (int) Math.ceil(data.getRows() / (double) rbmEdgeLength);
        this.originalData = new FloatMatrix(resizeBiLinear(data.toArray2(), pictureWidth, pictureHeight));
        this.originalImage = dataToImage(this.originalData);
    }

    @Override
    public Dimension getPreferredSize() {
        return new Dimension(900, 600);
    }

    private int[] convertGrayScalePicture(float[] values) {
        int[] result = new int[values.length];
        int lum;
        for (int i = 0; i < result.length; i++) {
            lum = (int)(255 * values[i]);
            if(lum > 255) lum = 255;
            if(lum < 0) lum = 0;
            result[i] = 0xFF000000 | lum << 16 | lum << 8 | lum;
        }

        return result;

    }
    private BufferedImage dataToImage(FloatMatrix data) {
        BufferedImage result = new BufferedImage(data.getColumns(), data.getRows(), BufferedImage.TYPE_INT_ARGB);
        result.setRGB(0,0, data.getColumns(), data.getRows(), convertGrayScalePicture(data.data),0, data.getColumns());
        return result;
    }


    private static float[][] resizeBiLinear(float[][] data, int newWidth, int newHeight) {

        float[][] temp = new float[newHeight][newWidth] ;
        int x, y;
        final double wRatio = ((double)(data[0].length - 1)) / newWidth ;
        final double hRatio = ((double)(data.length - 1)) / newHeight ;
        double tWidth, tHeight;
        for (int i = 0; i < newHeight; i++) {
            y = (int)(hRatio * i);
            tHeight = (hRatio * i) - y;
            double tHeightMinusOne = 1 - tHeight;

            for (int j = 0; j < newWidth; j++) {
                x = (int)(wRatio * j);
                tWidth = (wRatio * j) - x ;
                double tWidthMinusOne = 1 - tWidth;
                temp[i][j] = (float)(
                        data[y][x] * tWidthMinusOne * tHeightMinusOne +  data[y][x + 1] * tWidth * tHeightMinusOne +
                                data[y + 1][x] * tHeight * tWidthMinusOne   +  data[y + 1][x + 1] * tWidth * tHeight);
            }
        }
        return temp;
    }
    @Override
    public void update(RBMInfoPackage pack) {
        Dimension size = getPreferredSize();
        Graphics g = getGraphics();
        this.info = pack;
        paintImmediately(0, 0, size.width, size.height);
    }

    @Override
    public void paint(Graphics graphics) {
        super.paint(graphics);
        Graphics2D g = (Graphics2D)graphics;
        g.clearRect(0, 0, getPreferredSize().width, getPreferredSize().height);
        g.setColor(Color.black);
        g.fillRect(0,0, getPreferredSize().width, getPreferredSize().height);
        if(info != null) {

            FloatMatrix result = new FloatMatrix(originalData.getRows(), originalData.getColumns());
            FloatMatrix weights = new FloatMatrix(info.getWeights());
            RBM rbm = new RBM(weights);

            for (int i = 0; i <= originalData.getRows() - rbmEdgeLength; i += rbmEdgeLength) {
                for (int j = 0; j <= originalData.getColumns() - rbmEdgeLength; j += rbmEdgeLength) {
                    FloatMatrix patch = originalData.getRange(i, i + rbmEdgeLength, j, j + rbmEdgeLength);
                    float[][] hiddenBatch =  rbm.getHidden(new float[][]{patch.toArray()});
                    float[][] visibleBatch = rbm.getVisible(hiddenBatch);
                    result.put(new FloatMatrix(rbmEdgeLength, rbmEdgeLength,visibleBatch[0]), i, j);

                }
            }
            BufferedImage recon = dataToImage(result);
            g.drawImage(recon,0 , 0, 600, 600, null);
            g.setColor(Color.white);
            g.drawString("Error: " + info.getError() * 255, 650, 20);
            g.drawString("Epochs: " + info.getEpochs(), 650, 40);
            g.drawString("Features: " + weights.getColumns(), 650, 60);
        }
    }
}
