/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package berlin.iconn.projects.segmentation;

import berlin.iconn.rbm.DataConverter;

import java.awt.Dimension;
import java.awt.GridLayout;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.image.BufferedImage;
import javax.swing.*;

/**
 * @author christoph
 */
public class StackVisualization extends JComponent implements MouseListener {
    private final int compWidth = 1200;
    private final int compHeight = 600;

    private final int imageWidth;
    private final int imageHeight;

    private final int batchOffset;

    private final RBMSegmentationStack stack;
    private final float[] originalImage;
    private final float[] labelImage;
    private final float[] stackImage;
    private final String[] classes;
    private final boolean isRGB;

    private ImageComponent imgComp;


    
    RGB[] colors = new RGB[]{
            new RGB(104, 104, 3),
            new RGB(33, 44, 77),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB(),
            new RGB()
    };
    InteractiveKeyVisualisation itable;

    public StackVisualization(RBMSegmentationStack stack, float[] oi, int[] li, String[] classes, float minData, boolean isRGB, int batchOffset) {
        this.stack = stack;
        this.originalImage = oi;
        this.labelImage = labelDataToImage(li);
        this.classes = classes;

        // greyscale not supported yet
        this.isRGB = isRGB;
        this.batchOffset = batchOffset;

        this.setPreferredSize(new Dimension(compWidth, compHeight));
        this.setLayout(new GridLayout(1, 2));

        BufferedImage bi = DataConverter.pixelDataToImage(originalImage, minData, isRGB);
        this.imageWidth = bi.getWidth();
        this.imageHeight = bi.getHeight();

        this.imgComp = new ImageComponent(bi);
        this.add(imgComp);

        BufferedImage bi2 = DataConverter.pixelDataToImage(labelImage, minData, isRGB);
        this.add(new ImageComponent(bi2));

        stackImage = labelDataToImage(createStackImage(stack));
        BufferedImage bi3 = DataConverter.pixelDataToImage(stackImage, minData, isRGB);
        this.add(new ImageComponent(bi3));

        itable = new InteractiveKeyVisualisation();

        for (int c = 0; c < classes.length; c++) {
            itable.addRow(classes[c], colors[c], 0.0f);
        }
        /*
        itable.setValueAt(23, 0, 2);
        itable.setValueAt(23, 0, 3);
        */
        this.add(itable);

        this.repaint();
        imgComp.addMouseListener(this);
    }

    private void updateRow(int i, float p) {
        itable.setValueAt((int) (p * 100), i, 2);
        itable.setValueAt(((int) Math.round(p * 100)) + "%", i, 3);
    }

    private int[] createStackImage(RBMSegmentationStack s) {
        int[] tmpImage = new int[originalImage.length / 3];


        for (int j = batchOffset; j < imageWidth - batchOffset; j++) {
            for (int i = batchOffset; i < imageHeight - batchOffset; i++) {


                int batchSize = 2 * batchOffset + 1;
                float[] batch = new float[batchSize * batchSize * 3];

                int pos = 0;
                for (int y = i - batchOffset; y < i + batchOffset; y++) {
                    for (int x = j - batchOffset; x < j + batchOffset; x++) {

                        int arrayPos = y * imageWidth * 3 + x * 3;
                        float r = originalImage[arrayPos];
                        float g = originalImage[arrayPos + 1];
                        float b = originalImage[arrayPos + 2];

                        batch[pos++] = r;
                        batch[pos++] = g;
                        batch[pos++] = b;
                    }
                }

                float[] labelReconstruct = stack.reconstructLabel(batch, this.classes.length);

                String maxClass = "";
                float maxValue = 0.0f;
                int classLabel = 0;

                for (int c = 0; c < classes.length; c++) {
                    if (labelReconstruct[c] > maxValue) {
                        maxValue = labelReconstruct[c];
                        maxClass = classes[c];
                        classLabel = c;
                    }
                }
                tmpImage[i + j * imageWidth] = classLabel;
                //System.out.println(maxClass + ": " + maxValue);
            }
        }
        return tmpImage;
    }

    private float[] labelDataToImage(int[] im) {
        float[] tmpImage = new float[im.length * 3];

        int t = 0;
        for (int i = 0; i < im.length * 3; i += 3) {
            int m = im[t];
            tmpImage[i] = colors[m].nr;
            tmpImage[i + 1] = colors[m].ng;
            tmpImage[i + 2] = colors[m].nb;
            t++;
        }
        return tmpImage;
    }

    @Override
    public void mouseClicked(MouseEvent e) {
        int xMouse = e.getX();
        int yMouse = e.getY();

        if (xMouse > imageWidth - batchOffset || xMouse < batchOffset || yMouse > imageHeight - batchOffset || yMouse < batchOffset) {
            return;
        }

        int batchSize = 2 * batchOffset + 1;
        float[] batch = new float[batchSize * batchSize * 3];

        int pos = 0;
        for (int y = yMouse - batchOffset; y < yMouse + batchOffset; y++) {
            for (int x = xMouse - batchOffset; x < xMouse + batchOffset; x++) {

                int arrayPos = y * imageWidth * 3 + x * 3;
                float r = originalImage[arrayPos];
                float g = originalImage[arrayPos + 1];
                float b = originalImage[arrayPos + 2];

                batch[pos++] = r;
                batch[pos++] = g;
                batch[pos++] = b;
            }
        }

        float[] labelReconstruct = stack.reconstructLabel(batch, this.classes.length);

        String maxClass = "";
        float maxValue = 0.0f;

        for (int c = 0; c < classes.length; c++) {
            updateRow(c, labelReconstruct[c]);
           // System.out.println(c + ": " + labelReconstruct[c]);
            //
            //
            if (labelReconstruct[c] > maxValue) {
                maxValue = labelReconstruct[c];
                maxClass = classes[c];
            }
        }

        System.out.println(maxClass + ": " + maxValue);
    }

    @Override
    public void mousePressed(MouseEvent e) {
        //throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void mouseReleased(MouseEvent e) {
        //throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void mouseEntered(MouseEvent e) {
        //throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void mouseExited(MouseEvent e) {
        //throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }


}
