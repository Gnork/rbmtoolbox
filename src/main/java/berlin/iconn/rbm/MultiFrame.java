package berlin.iconn.rbm;

import javax.swing.*;
import java.awt.*;
import java.util.Arrays;

/**
 * Created by Moritz on 5/18/2014.
 */
public class MultiFrame extends JFrame {
    public MultiFrame(JComponent[] pictures) {

        int height = 0;
        int width = 0;

        for (int i = 0; i < pictures.length; i++) {
            final Dimension preferredSize = pictures[i].getPreferredSize();
            final double preferredSizeWidth = preferredSize.getWidth();
            if(preferredSizeWidth > width) {
                width += (int) preferredSizeWidth;
            }
            final double preferredSizeHeight = preferredSize.getHeight();
            if(preferredSizeHeight > height) {
                height = (int) preferredSizeHeight;
            }
        }
        this.setLayout(new FlowLayout());
        this.setSize(new Dimension(width , height));
        this.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        Arrays.asList(pictures).stream().forEach(picture -> this.add(picture));
        this.pack();
        this.setVisible(true);
    }
}
