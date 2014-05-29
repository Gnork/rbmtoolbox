/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package berlin.iconn.projects.segmentation;

import berlin.iconn.rbm.DataConverter;
import berlin.iconn.rbm.RBM;
import java.awt.Dimension;
import java.awt.GridLayout;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.image.BufferedImage;
import javax.swing.JComponent;

/**
 *
 * @author christoph
 */
public class StackVisualization extends JComponent implements MouseListener{
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
    
    public StackVisualization(RBMSegmentationStack stack, float[] image, String[] classes, float minData, boolean isRGB, int batchOffset){
        this.stack = stack;
        this.image = image;
        this.classes = classes;
        // greyscale not supported yet
        this.isRGB = isRGB;
        this.batchOffset = batchOffset;
        
        this.setPreferredSize(new Dimension(compWidth, compHeight));
        this.setLayout(new GridLayout(1,2));
        
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
        
        if(xMouse > imageWidth - batchOffset || xMouse < batchOffset || yMouse > imageHeight - batchOffset || yMouse < batchOffset) {
            return;
        }
        
        int batchSize = 2 * batchOffset + 1;
        float[] batch = new float[batchSize * batchSize * 3];
        
        int pos = 0;
        for(int y = yMouse - batchOffset; y < yMouse + batchOffset; y++) {
            for(int x = xMouse - batchOffset; x < xMouse + batchOffset; x++) {
                
                int arrayPos = y * imageWidth * 3 + x*3;
                float r = image[arrayPos];
                float g = image[arrayPos+1];
                float b = image[arrayPos+2];
                
                batch[pos++] = r;
                batch[pos++] = g;
                batch[pos++] = b;
            }
        }
        
        float[] labelReconstruct = stack.reconstructLabel(batch, this.classes.length);
        
        String maxClass = "";
        float maxValue = 0.0f;
        
        for(int c = 0; c < classes.length; c++) {
            if(labelReconstruct[c] > maxValue) {
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
