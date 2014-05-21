package berlin.iconn.rbm.enhancements.visualizations;

import berlin.iconn.rbm.enhancements.RBMInfoPackage;
import com.googlecode.client.Line;

import javax.swing.*;
import java.awt.*;
import java.awt.geom.Line2D;
import java.util.ArrayList;

/**
 * Created by Moritz on 5/18/2014.
 */

public class ErrorDataVisualization extends JComponent implements IVisualizeObserver {

    private RBMInfoPackage info;
    private final int width, height;

    private float max;

    private final double margin = 20;

    private class GraphData {
        public float error, epoch;

        public GraphData(float er, float ep) {
            ;
            this.error = er;
            this.epoch = ep;
        }
    }

    private ArrayList<GraphData> graphData;

    public ErrorDataVisualization() {
        this.width = 400;
        this.height = 200;
        graphData = new ArrayList<>();
    }

    public Dimension getPreferredSize() {
        return new Dimension(width, height);
    }

    @Override
    public void update(RBMInfoPackage pack) {
        this.info = pack;
        paintImmediately(0, 0, width, height);
    }

    @Override
    public void paint(Graphics graphics) {


        Graphics2D g = (Graphics2D) graphics;
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.clearRect(0, 0, getPreferredSize().width, getPreferredSize().height);
        g.setColor(Color.white);
        g.fillRect(0, 0, getPreferredSize().width, getPreferredSize().height);

        if (info != null) {

            g.setColor(Color.blue);
            float error = info.getError() * 255;
            g.drawString("Error: " + error, 0, 20);
            g.drawString("Epochs: " + info.getEpochs(), 0, 40);

            graphData.add(new GraphData(error, info.getEpochs()));

            if (graphData.size() == 1)
                max = graphData.get(0).error;

            double margin = 20;

            double lw = 15;
            double wordspaceY = 50;
            double gOrigXY = margin;
            double newH = height - 2 * margin;
            double newW = width - 2 * margin;
            double tmpMax = Math.ceil(max);
            double step = newH / tmpMax;
            double fontH = 10;

            g.setColor(new Color(128, 128, 128));
            for (int i = (int) tmpMax; i >= 0; i--) {
                double tmpY = gOrigXY + step * i;

                g.drawString(""+ (tmpMax - (i)),(int) gOrigXY , (int)(fontH/2+gOrigXY+(newH)*(i)/tmpMax));
                //g.drawString("Epochs: " + graphData.get(i).error, 20, (int) y1);

                g.draw(new Line2D.Double(gOrigXY + wordspaceY, tmpY, gOrigXY + newW, tmpY));
            }


            g.setColor(new Color(0, 143, 255));
            for (int i = 0; i < graphData.size() - 1; i++) {


                double y1 =  gOrigXY + newH - step *graphData.get(i).error;
                double y2 =  gOrigXY + newH - step *graphData.get(i+1).error;


                //g.drawString("Epochs: " + graphData.get(i).error, 20, (int) y1);
                g.draw(new Line2D.Double(gOrigXY+wordspaceY+ (lw *i), y1, gOrigXY+wordspaceY+ (lw *(i+1)), y2));
            }
        }
    }
}
