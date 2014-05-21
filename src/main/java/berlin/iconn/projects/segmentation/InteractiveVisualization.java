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
import java.awt.image.BufferedImage;
import javax.swing.JComponent;

/**
 *
 * @author christoph
 */
public class InteractiveVisualization extends JComponent implements MouseListener{
    private final int compWidth = 800;
    private final int compHeight = 600;
    
    private final RBM rbm;
    private final float[] image;
    private final String[] classes;
    private final boolean isRGB;
    
    private ImageComponent imgComp;
    
    public InteractiveVisualization(RBM rbm, float[] image, String[] classes, float minData, boolean isRGB){
        this.rbm = rbm;
        this.image = image;
        this.classes = classes;
        this.isRGB = isRGB;
        
        this.setPreferredSize(new Dimension(compWidth, compHeight));
        this.setLayout(new GridLayout(1,2));
        
        BufferedImage bi = DataConverter.pixelDataToImage(image, minData, isRGB);
        
        this.imgComp = new ImageComponent(bi);
        this.add(imgComp);
        this.repaint();
    }

    @Override
    public void mouseClicked(MouseEvent e) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void mousePressed(MouseEvent e) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void mouseReleased(MouseEvent e) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void mouseEntered(MouseEvent e) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void mouseExited(MouseEvent e) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    
}
