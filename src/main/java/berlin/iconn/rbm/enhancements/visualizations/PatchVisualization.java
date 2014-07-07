package berlin.iconn.rbm.enhancements.visualizations;

import berlin.iconn.persistence.InOutOperations;
import berlin.iconn.rbm.DataConverter;
import berlin.iconn.rbm.IRBM;
import berlin.iconn.rbm.RBM;
import berlin.iconn.rbm.enhancements.RBMInfoPackage;
import org.jblas.FloatMatrix;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;

/**
 * Created by Moritz on 7/6/2014.
 */
public class PatchVisualization extends JComponent implements IVisualizeObserver {
    private RBMInfoPackage info;
    private final int patchSize;
    private int imageSize;
    private final FloatMatrix patches;
    private final IRBM rbm;

    public PatchVisualization(int patchSize, int imageSize, BufferedImage image, IRBM rbm) {
        this.patchSize = patchSize;
        int sizeRemainder = imageSize % patchSize;
        BufferedImage img = image;
        this.imageSize = imageSize;
        if(sizeRemainder != 0) {
            this.imageSize = (imageSize / patchSize + 1) * patchSize;
            img = resize(img, this.imageSize);
        }
        int count = this.imageSize / patchSize;
        float[][] patches = new float[count * count][];
        for (int i = 0; i < count; i++) {
            for (int j = 0; j < count; j++) {
                BufferedImage part = img.getSubimage(i * patchSize, j * patchSize, patchSize, patchSize);
                patches[i * count + j] = DataConverter.processPixelData(part, patchSize, false, false, 0.0f, 1.0f, true);
            }
        }
        this.patches = new FloatMatrix(patches);


        this.imageSize = imageSize;
        this.rbm = rbm;
    }

    private BufferedImage resize(BufferedImage img, int newSize) {
        int w = img.getWidth();
        int h = img.getHeight();
        BufferedImage result = new BufferedImage(newSize, newSize, img.getType());
        Graphics2D g = result.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(img, 0, 0, newSize, newSize, 0, 0, w, h, null);
        g.dispose();
        return result;
    }

    @Override
    public Dimension getPreferredSize() {
        return new Dimension(900, 600);
    }

    private int[] convertGrayScalePicture(float[] values) {
        int[] result = new int[values.length];
        int lum;
        for (int i = 0; i < result.length; i++) {
            lum = convertValue(values[i]);
            result[i] = 0xFF000000 | lum << 16 | lum << 8 | lum;
        }

        return result;
    }

    private int[] convertRGBPicture(float[] values) {
        int[] result = new int[values.length / 3];

        for (int i = 0; i < result.length; i++) {
            int index = i * 3;
            int r = convertValue(values[index]);
            int g = convertValue(values[index + 1]);
            int b = convertValue(values[index + 2]);
            result[i] = 0xFF000000 | r << 16 | g << 8 | b;
        }

        return result;
    }

    private int convertValue(float value) {
        int lum;
        lum = (int)(255 * value);
        if(lum > 255) lum = 255;
        if(lum < 0) lum = 0;
        return lum;
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
            g.setColor(Color.white);
            g.drawString("Error: " + info.getError() * 255, 650, 20);
            g.drawString("Epochs: " + info.getEpochs(), 650, 40);
            g.drawString("Features: " + info.getWeights()[0].length, 650, 60);
        }
    }
}
