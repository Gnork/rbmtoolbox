/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package berlin.iconn.projects.segmentation;

import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import javax.swing.JComponent;

/**
 *
 * @author christoph
 */
public class ImageComponent extends JComponent {
    
    private BufferedImage image;
    
    public ImageComponent(BufferedImage image){
        this.image = image;
    }

    @Override
    public void paint(Graphics g) {
        super.paint(g);
        Graphics2D graphics = (Graphics2D)g;
        graphics.drawImage((Image) image,0,0,null);
    }
    
    
}
