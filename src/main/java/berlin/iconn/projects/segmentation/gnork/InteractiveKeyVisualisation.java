package berlin.iconn.projects.segmentation.gnork;

import javax.swing.*;
import javax.swing.table.DefaultTableCellRenderer;
import javax.swing.table.DefaultTableModel;
import java.awt.*;
import java.awt.image.BufferedImage;

/**
 * Created by G on 02.06.14.
 */
public class InteractiveKeyVisualisation extends JComponent {

    //txt.setHorizontalTextPosition(JLabel.CENTER);
    //txt.setFont(new Font("Serif", Font.PLAIN, 21));

    private JTable table;
    public InteractiveKeyVisualisation() {


        Object rowData[][] = {
        };

        Object columnNames[] = {
                "Class Name",
                "Color",
                "Probability of Region",
                "Probability of Region"
        };

        DefaultTableModel model = new DefaultTableModel(rowData, columnNames) {
            private static final long serialVersionUID = 1L;

            @Override
            public Class<?> getColumnClass(int column) {
                return getValueAt(0, column).getClass();
            }

            @Override
            public boolean isCellEditable(int row, int col) {
                return false;
            }
        };
        table = new JTable(model);

        table.getColumnModel().getColumn(2).setCellRenderer(new ProgressRenderer());
        table.getColumnModel().getColumn(1).setCellRenderer(new ColorRenderer());
        JScrollPane scrollPane = new JScrollPane(table);
        this.setLayout(new GridLayout(1, 0));
        this.add(scrollPane);
    }

    public void addRow(String n, RGB rgb, float propability){
        int t = (int)propability;
        DefaultTableModel model = (DefaultTableModel) table.getModel();
        model.addRow(new Object[]{n, rgb, t, t});
    }

    public void setValueAt(Object v, int y, int x){
        table.setValueAt(v, y, x);
    }

    class ProgressRenderer extends DefaultTableCellRenderer {

        private final JProgressBar b = new JProgressBar(0, 100);

        public ProgressRenderer() {
            super();
            setOpaque(true);
            b.setBorder(BorderFactory.createEmptyBorder(1, 1, 1, 1));
        }

        @Override
        public Component getTableCellRendererComponent(JTable table, Object value, boolean isSelected, boolean hasFocus, int row, int column) {
            Integer i = (Integer) value;
            String text = "Completed";
            if (i < 0) {
                text = "Error";
            } else if (i <= 100) {
                b.setValue(i);
                return b;
            }
            super.getTableCellRendererComponent(table, text, isSelected, hasFocus, row, column);
            return this;
        }
    }

    class ColorRenderer extends DefaultTableCellRenderer {
        public ColorRenderer() {
            super();
            setOpaque(true);
        }

        @Override
        public Component getTableCellRendererComponent(JTable table, Object value, boolean isSelected, boolean hasFocus, int row, int column) {
            RGB i = (RGB) value;

            BufferedImage bi = new BufferedImage(20, 20, BufferedImage.TYPE_INT_ARGB);
            Graphics2D    graphics = bi.createGraphics();

            graphics.setPaint ( new Color ( i.r, i.g, i.b) );
            graphics.fillRect ( 0, 0, bi.getWidth(), bi.getHeight() );

            return new ImageComponent(bi);
        }
    }
}
