package berlin.iconn.rbm.enhancements.visualizations;

import berlin.iconn.rbm.enhancements.RBMInfoPackage;

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

    private class GraphData {
        public float error, epoch;

        public GraphData(float er, float ep) {
            this.epoch = ep;
            this.error = er;
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
        g.clearRect(0, 0, getPreferredSize().width, getPreferredSize().height);
        g.setColor(Color.black);
        g.fillRect(0, 0, getPreferredSize().width, getPreferredSize().height);

        if (info != null) {

            g.setColor(Color.white);
            g.drawString("Error: " + info.getError() * 255, 0, 20);
            g.drawString("Epochs: " + info.getEpochs(), 0, 40);

            graphData.add(new GraphData(info.getError(), info.getEpochs()));

            if (graphData.size() == 1)
                max = graphData.get(0).error;

            int lum = (int) (255);
            g.setColor(new Color(lum, lum, lum));

            double w = 3.0;

            for (int i = 0; i < graphData.size() - 1; i++) {
                double y1 = graphData.get(i).error / max * height;
                double y2 = graphData.get(i + 1).error / max * height;
                g.draw(new Line2D.Double(i * w, height - y1, (i + 1) * w, height - y2));
            }
        }
    }
}
