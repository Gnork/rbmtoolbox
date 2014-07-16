package berlin.iconn.projects.set;

import berlin.iconn.persistence.InOutOperations;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

/**
 * Created by Moritz on 7/11/2014.
 */
public class SplitAndResizeMain {

    public static void main(String[] args) {
        int count = 0;
        File[] images = InOutOperations.getImageFiles(new File("Data/cross"));
        for (int i = 0; i < images.length; i++) {
            BufferedImage image = null;
            try {
                image = ImageIO.read(images[i]);
            } catch (IOException e) {
                e.printStackTrace();
            }
            int height = 512;
            float ratio = height / (float) image.getHeight();
            int width = (int) Math.ceil(ratio * image.getWidth());

            BufferedImage resize = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
            Graphics2D g = resize.createGraphics();
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                    RenderingHints.VALUE_INTERPOLATION_BILINEAR);
            g.drawImage(image, 0,0, width, height, 0, 0, image.getWidth(), image.getHeight(), null);
            g.dispose();
            if(width / (float) height > 1.5f) {
                int partX = 0;
                do {
                    String name = String.format("%04d", count) + ".jpg";
                    BufferedImage part = resize.getSubimage(partX, 0, height, height);
                    try {
                        ImageIO.write(part, "jpg", new File("Data/result/" + name));
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    partX += height;
                    count++;
                } while (partX + height < resize.getWidth());
            }
            String name = String.format("%04d", count) + ".jpg";
            BufferedImage part = resize.getSubimage(resize.getWidth() - height, 0, height, height);
            try {
                ImageIO.write(part, "jpg", new File("Data/result/" + name));
            } catch (IOException e) {
                e.printStackTrace();
            }
            count++;
        }
    }
}
