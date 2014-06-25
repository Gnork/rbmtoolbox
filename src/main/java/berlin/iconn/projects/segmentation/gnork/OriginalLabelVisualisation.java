package berlin.iconn.projects.segmentation.gnork;

import berlin.iconn.rbm.DataConverter;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.image.BufferedImage;

/**
 * Created by G on 01.06.14.
 */
public class OriginalLabelVisualisation extends JComponent implements MouseListener {
    private final int compWidth = 800;
    private final int compHeight = 600;

    private final int imageWidth;
    private final int imageHeight;

    private final int batchOffset;

    private final RBMSegmentationStack stack;
    private final float[] image;
    private final String[] classes;
    private final boolean isRGB;

    private ImageComponent imgComp;

    public OriginalLabelVisualisation(RBMSegmentationStack s, int[] im, String[] c, float minData, boolean is, int bo) {
        stack = s;

        image = new float[im.length * 3];

        int t = 0;
        for (int i = 0; i < im.length * 3 ; i += 3) {
            float m = 50 / (float) im[t];
            image[i] = m;
            image[i + 1] = m;
            image[i + 2] = m;
            t++;
        }

        classes = c;

        isRGB = is;
        this.batchOffset = bo;

        this.setPreferredSize(new Dimension(compWidth, compHeight));
        this.setLayout(new GridLayout(1, 2));

        BufferedImage bi = DataConverter.pixelDataToImage(image, minData, isRGB);
        this.imageWidth = bi.getWidth();
        this.imageHeight = bi.getHeight();

        this.imgComp = new ImageComponent(bi);
        this.add(imgComp);
        this.repaint();

        // Reconstruction

        imgComp.addMouseListener(this);
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
                float r = image[arrayPos];
                float g = image[arrayPos + 1];
                float b = image[arrayPos + 2];

                batch[pos++] = r;
                batch[pos++] = g;
                batch[pos++] = b;
            }
        }

        float[] labelReconstruct = stack.reconstructLabel(batch, this.classes.length);

        String maxClass = "";
        float maxValue = 0.0f;

        for (int c = 0; c < classes.length; c++) {
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
