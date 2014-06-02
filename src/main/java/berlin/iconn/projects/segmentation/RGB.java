package berlin.iconn.projects.segmentation;

/**
 * Created by G on 02.06.14.
 */
public class RGB {
    int r, g, b;
    float nr, ng, nb;

    RGB(int rr, int gg, int bb) {
        nr = rr / 255.0f;
        ng = gg / 255.0f;
        nb = bb / 255.0f;
        r = rr;
        g = gg;
        b = bb;
    }

    RGB() {
        nr = (float) Math.random();
        ng = (float) Math.random();
        nb = (float) Math.random();

        r = (int) (nr * 255);
        g = (int) (ng * 255);
        b = (int) (nb * 255);
    }
}