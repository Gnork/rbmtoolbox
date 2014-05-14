package berlin.iconn.scanpicture;

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
        this.originalData = resizeBiLinear(data, pictureWidth, pictureHeight);
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

    private FloatMatrix resizeBiLinear(FloatMatrix m, int newWidth, int newHeight) {
        float[][] original = m.toArray2();
        float[][] result =  new float[newHeight][newWidth];
        double hResize = (m.getRows() - 1) / (newHeight - 1);
        double wResize = (m.getRows() - 1) / (newHeight - 1);

        for (int i = 0; i < newHeight - 1; i++) {

            double indexH = i * hResize;
            final int floorH = (int)indexH;
            final int ceilH = floorH + 1;
            final double tH = indexH - floorH;

            for (int j = 0; j < newWidth - 1; j++) {
                final double indexW = j * wResize;
                final int floorW = (int)indexW;
                final int ceilW = floorW + 1;
                final double tW = indexW - floorW;
                final double minOneTW = 1.0 - tW;
                result[i][j] = (float)(
                        (original[floorH][floorW] * minOneTW + original[floorH][ceilW] * tW) * (1.0 - tH) +
                        (original[ceilH] [floorW] * minOneTW + original[ceilH] [ceilW] * tW) * tH);
            }
        }
        return new FloatMatrix(result);
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
                    FloatMatrix batch = originalData.getRange(i, i + rbmEdgeLength, j, j + rbmEdgeLength);
                    float[][] hiddenBatch =  rbm.getHidden(new float[][]{batch.toArray()});
                    float[][] visibleBatch = rbm.getVisible(hiddenBatch);
                    result.put(new FloatMatrix(rbmEdgeLength, rbmEdgeLength,visibleBatch[0]), i, j);

                }
            }
            BufferedImage recon = dataToImage(result);
            g.drawImage(recon,0 , 0, 600, 600, null);
            g.setColor(Color.white);
//            g.drawString("Error: " + info.getError() * 255, 650, 20);
            g.drawString("Epochs: " + info.getEpochs(), 650, 40);
            g.drawString("Features: " + weights.getColumns(), 650, 60);
        }
    }
}
