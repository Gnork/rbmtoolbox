package berlin.iconn.projects.facerepair;

import berlin.iconn.persistence.InOutOperations;
import berlin.iconn.rbm.DataConverter;
import berlin.iconn.rbm.RBM;
import ij.*;
import ij.gui.ImageCanvas;
import ij.gui.Roi;
import ij.process.ImageProcessor;
import org.jblas.FloatMatrix;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Created by GDur on 26.06.14.
 */
public class MainInteractive {

    private static final int EDGE_LENGTH = 64;

    private static final String IMAGES_TRAINED = "D:\\image_sets\\rbm_face_images_png\\1000_images_trained";
    private static final String IMAGES_TRAINED_INCOMPLETE = "D:\\image_sets\\rbm_face_images_png\\1000_images_trained_incomplete_gray";
    private static final String IMAGES_NOT_TRAINED = "D:\\image_sets\\rbm_face_images_png\\1000_images_not_trained";
    private static final String IMAGES_NOT_TRAINED_INCOMPLETE = "D:\\image_sets\\rbm_face_images_png\\1000_images_not_trained_incomplete_gray";

    private static final String RBM1_WEIGHTS = "Output/SimpleWeights/WildFaces_64x64_rgb_1kh_2700it.dat";

    static RBM[] rbms;

    static ImagePlus imp;

    public static void main(String args[]) {
        String file = "D:\\RootWorkspace\\furry-avenger\\rbm_face_images_png\\training_set\\Abbas_Kiarostami_0001.png";
        //file = "D:\\RootWorkspace\\furry-avenger\\rbm_face_images_png\\training_set\\Abdul_Majeed_Shobokshi_0001.png";

        ImagePlus imp = IJ.openImage(file);
        ImageCanvas ic = new ImageCanvas(imp);
        ImagePlus imp2 = IJ.openImage(file);
        ImageCanvas ic2 = new ImageCanvas(imp2);

        JFrame frame2 = new JFrame("Original");
        frame2.add(ic);

        JFrame frame3 = new JFrame("Reconstructed");
        frame3.add(ic2);

        //imp2.getProcessor().setColor(new Color(169, 168, 172));
        //imp2.getProcessor().fill();

        frame2.pack();
        frame2.setLocationRelativeTo(null);
        frame2.setVisible(true);
        // frame3.pack();
        //frame3.setLocationRelativeTo(null);
        //frame3.setVisible(true);
        FloatMatrix rbm1Weights;
        try {
            rbm1Weights = new FloatMatrix(InOutOperations.loadSimpleWeights(new File(RBM1_WEIGHTS)));
        } catch (IOException | ClassNotFoundException ex) {
            Logger.getLogger(MainTest.class.getName()).log(Level.SEVERE, null, ex);
            return;
        }
        RBM rbm1 = new RBM(rbm1Weights);
        rbms = new RBM[]{rbm1};
        System.out.println("RBMs loaded");


        ImageJ test = new ImageJ();
        IJ.open(file);
        JFrame frame = new JFrame("reconstruct");

        JButton button = new JButton("reconstruct");
        button.addActionListener(ae -> {
            crop();
        });
        frame.add(button);
        /*
        JButton button2 = new JButton("open");
        button2.addActionListener(ae -> {
            final ImagePlus original = WindowManager.getCurrentImage();
            System.out.println( WindowManager.getCurrentImage().getTitle());
            crop();
        });
        frame.add(button2);
*/
        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }

    private static float[] toFloatArray(ImagePlus from) {
        BufferedImage t = from.getBufferedImage();
        float[] to = new float[t.getHeight() * t.getWidth() * 3];

        int xy = 0;
        for (int y = 0; y < t.getHeight(); y++) {
            for (int x = 0; x < t.getWidth(); x++) {
                int clr = t.getRGB(x, y);

                int red = (clr & 0x00ff0000) >> 16;
                to[xy] = (red) / 255.0f;
                xy++;

                int green = (clr & 0x0000ff00) >> 8;
                to[xy] = (green) / 255.0f;
                xy++;

                int blue = clr & 0x000000ff;
                to[xy] = (blue) / 255.0f;
                xy++;
            }
        }
        return to;
    }

    private static void reconstruction(RBM[] rbms, int edgeLength, ImagePlus testData, ImagePlus compareData, String testName) throws IOException {
        System.out.println("Starting Test: " + testName);
        float[][] testDataFloat = new float[1][];
        testDataFloat[0] = toFloatArray(testData);

        float[][] compareDataFloat = new float[1][];
        compareDataFloat[0] = toFloatArray(compareData);

        FloatMatrix reconData = new FloatMatrix(testDataFloat);
        for (int i = 0; i < rbms.length; ++i) {
            reconData = rbms[i].getHidden(reconData);
        }
        for (int i = rbms.length - 1; i >= 0; --i) {
            reconData = rbms[i].getVisible(reconData);
        }

        float[][] reconDataFloat = reconData.toArray2();

        compareArraysForError(reconDataFloat[0], compareDataFloat[0], testName);
    }

    private static void compareArraysForError(float[] reconData, float[] compareData, String testName) throws IOException {
        String dirString = "Output/" + testName;
        InOutOperations.mkdir(dirString);

        FileWriter writer = new FileWriter(dirString + "/results.txt");

        String newLine = System.getProperty("line.separator");

        float imageError = 0.0f;
        for (int j = 0; j < reconData.length; ++j) {
            imageError += Math.abs(reconData[j] - compareData[j]);
        }
        imageError /= reconData.length;
        String errorOut = "error: " + imageError;

        System.out.println(errorOut);
        writer.write(errorOut + newLine);

        BufferedImage bi = DataConverter.pixelDataToImage(reconData, 0.0f, true);
        File imageOut = new File(dirString + "/recon.png");
        ImageIO.write(bi, "png", imageOut);

        writer.close();
        IJ.open(dirString + "/recon.png");
    }

    private static void crop() {
        final ImagePlus original = WindowManager.getCurrentImage();

        try {
            reconstruction(rbms, EDGE_LENGTH, original, original, "IMAGES_TRAINED");
        } catch (IOException ex) {
            Logger.getLogger(MainTest.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}