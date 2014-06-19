package berlin.iconn.rbm.enhancements.visualizations;

import berlin.iconn.rbm.RBM;
import berlin.iconn.rbm.enhancements.RBMInfoPackage;
import org.jblas.FloatMatrix;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;

/**
 * Created by Moritz on 5/18/2014.
 */
public class FeatureDataVisualization extends JComponent implements IVisualizeObserver {

    private final int height;
    private RBMInfoPackage info;
    private final int sizeMultiplier;
    private final int picturesRowSize;
    private final int edgeLength;
    private final int width;
    private float[][] data;
    private final BufferedImage[] dataImages;

    public FeatureDataVisualization(int sizeMultiplier, int picturesRowCount, int edgeLength, float[][] data) {
        this.sizeMultiplier = sizeMultiplier;
        this.picturesRowSize = picturesRowCount;
        this.edgeLength = edgeLength;
        this.width = edgeLength * (picturesRowSize + 1) * sizeMultiplier;
        this.height = edgeLength * (picturesRowSize + 1) * sizeMultiplier + 65;
        this.data = getData(data);
        this.dataImages = createBufferedImages(this.data);

    }

    private BufferedImage[] createBufferedImages(float[][] data) {
        BufferedImage[] result = new BufferedImage[this.data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = new BufferedImage(edgeLength, edgeLength,BufferedImage.TYPE_INT_ARGB);
            result[i].setRGB(0, 0, edgeLength, edgeLength, convertPicture(data[i]), 0, edgeLength);
        }
        return result;
    }

    private float[][] getData(float[][] data) {
        if(picturesRowSize >= data.length) {
            return data;
        } else {
            float[][] newData = new float[picturesRowSize][];
            Random random = new Random();
            HashSet<Integer> selection = new HashSet<Integer>();
            while(selection.size() < picturesRowSize) {
                selection.add(random.nextInt(data.length));
            }
            Integer[] selectionArray = selection.toArray(new Integer[0]);
            Arrays.sort(selectionArray);
            for (int i = 0; i < newData.length; i++) {
                newData[i] = data[selectionArray[i].intValue()];
            }
            return newData;
        }
    }

    public Dimension getPreferredSize() {

        return new Dimension(width, height);
    }

    @Override
    public void update(RBMInfoPackage pack) {
        this.info = pack;

        Graphics g = getGraphics();

        paintImmediately(0, 0, width, height);
    }

    @Override
    public void paint(Graphics graphics) {

        Graphics2D g = (Graphics2D)graphics;
        g.clearRect(0, 0, getPreferredSize().width, getPreferredSize().height);
        g.setColor(Color.black);
        g.fillRect(0,0, getPreferredSize().width, getPreferredSize().height);

        int drawSize = edgeLength * sizeMultiplier;
        if(info != null) {
            float[][] weights = new FloatMatrix(info.getWeights()).transpose().toArray2();


            g.setColor(Color.white);
            g.drawString("Error: " + info.getError() * 255, 0, 20);
            g.drawString("Epochs: " + info.getEpochs(), 0, 40);
            g.drawString("Features: " + weights.length, 0, 60);
            int length = weights.length;
            if(length > picturesRowSize) length = picturesRowSize;
            for (int i = 0; i < length ; i++) {

                BufferedImage bufferedImage = new BufferedImage(edgeLength, edgeLength,BufferedImage.TYPE_INT_ARGB);
                bufferedImage.setRGB(0, 0, edgeLength, edgeLength, convertWeights(weights[i]), 0, edgeLength);
                int x = drawSize * (i % picturesRowSize);
                int y = (drawSize * i) / width;
                g.drawImage(bufferedImage, (drawSize * 2) + x, 65 + y * drawSize, drawSize, drawSize, null);
            }

            RBM rbm = new RBM(new FloatMatrix(info.getWeights()));
            float[][] visible = rbm.getVisible(rbm.getHidden(data));
            BufferedImage[] reconstruction = createBufferedImages(visible);
            for (int i = 0; i < reconstruction.length; i++) {
                int y = drawSize * i;
                g.drawImage(reconstruction[i],  drawSize, drawSize + 65 + y, drawSize, drawSize, null);
            }
            float[][] hidden = rbm.getHidden(data);
            for (int i = 0; i < hidden.length; i++) {
                int y = (i + 1)* drawSize + 65;
                for (int j = 0; j < hidden[0].length; j++) {
                    int x = (j + 2) * drawSize;
                    int lum = (int)(hidden[i][j] * 255);
                    g.setColor(new Color(lum, lum, lum));
                    g.fillRect(x, y, drawSize, drawSize);
                }
            }

        }

        for (int i = 0; i < dataImages.length; i++) {
            int y = drawSize * i;
            g.drawImage(dataImages[i], 0, drawSize + 65 + y , drawSize, drawSize, null);
        }

    }

    private int[] convertPicture(float[] values) {
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

    private int[] convertWeights(float[] weight) {
        int[] result = new int[weight.length];
        int green;
        int red;
        for (int i = 0; i < result.length; i++) {
            red = 0;
            green =0;
            if(weight[i] < 0) {
                green = -(int)(255 * weight[i] * 1.0f);
                if(green > 255) green = 255;
            } else {
                red = (int)(255 * weight[i]  * 1.0f);
                if(red > 255) red = 255;
            }
            result[i] = 0xFF000000 | red << 16 | green << 8;
        }

        return result;
    }


}
