package berlin.iconn.projects.segmentation;

/**
 * Created by G on 02.06.14.
 */
public class RGB {
    int r, g, b;
    float nr, ng, nb;

    RGB(int rgb) {
        r = (rgb >>> 16) & 0xff;
        g = (rgb >>> 8) & 0xff;
        b = (rgb >>> 0) & 0xff;

        set();
    }

    public void set() {
        nr = r / 255.0f;
        ng = g / 255.0f;
        nb = b / 255.0f;
    }

    RGB(int rr, int gg, int bb) {
        r = rr;
        g = gg;
        b = bb;
        set();
    }

    RGB() {
        r = (int) (Math.random() * 255);
        g = (int) (Math.random() * 255);
        b = (int) (Math.random() * 255);
        set();
    }
}