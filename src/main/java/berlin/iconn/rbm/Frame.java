package berlin.iconn.rbm;

import javax.swing.*;

/**
 * Created by Moritz on 5/18/2014.
 */
public class Frame extends JFrame {
    public Frame(JComponent picture) {
        this.setSize(picture.getPreferredSize());
        this.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        this.add(picture);
        this.pack();
        this.setVisible(true);
    }
}
