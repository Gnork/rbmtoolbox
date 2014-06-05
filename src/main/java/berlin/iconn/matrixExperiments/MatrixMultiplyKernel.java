package berlin.iconn.matrixExperiments;

import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.RecursiveAction;


/**
 *
 * @author Moritz
 */
class MatrixMultiplyKernel extends RecursiveAction {

    static final int parallelThresHold = 0x40000;
    final int aRowStart;
    final int aRowEnd;
    final int bRowStart;
    final int bRowEnd;
    final float[][] a;
    final float[][] b;
    final LinkedBlockingQueue queue;

    public MatrixMultiplyKernel(LinkedBlockingQueue queue,
                                float[][] a, float[][] b) {
        this(queue, a, b, 0, a.length, 0, b.length);
    }

    public MatrixMultiplyKernel(
            LinkedBlockingQueue queue,
            float[][] a, float[][] b,
            int aRowStart, int aRowEnd, int bRowStart, int bRowEnd) {
        this.aRowStart = aRowStart;
        this.aRowEnd = aRowEnd;
        this.bRowStart = bRowStart;
        this.bRowEnd = bRowEnd;
        this.a = a;
        this.b = b;
        this.queue = queue;
    }

    @Override
    protected void compute() {
        int aRange = aRowEnd - aRowStart;
        int bRange = bRowEnd - bRowStart;
        if (aRange * bRange > parallelThresHold) {
            int split;
            if (aRange < bRange) {
                split = (bRowEnd + bRowStart) / 2;
                invokeAll(new MatrixMultiplyKernel(queue,
                                a, b,
                                aRowStart, aRowEnd, bRowStart, split),
                        new MatrixMultiplyKernel(queue,
                                a, b,
                                aRowStart, aRowEnd, split, bRowEnd));

            } else {
                split = (aRowEnd + aRowStart) / 2;
                invokeAll(new MatrixMultiplyKernel(queue,
                                a, b,
                                split, aRowEnd, bRowStart, bRowEnd),
                        new MatrixMultiplyKernel(queue,
                                a, b,
                                aRowStart, split, bRowStart, bRowEnd));
            }

        } else {
            float[][] c = new float[aRowEnd - aRowStart][b[0].length];
            mmul(a, b, c, aRowStart, aRowEnd, bRowStart, bRowEnd, aRowStart);
            queue.add(new OffsetArray(c, aRowStart));
        }
    }

    public static void mmul(float[][] a, float[][] b, float[][] c,
                            final int aRowStart, final int aRowEnd,
                            final int bRowStart, final int bRowEnd) {
        mmul(a, b, c, aRowStart, aRowEnd, bRowStart, bRowEnd, 0);
    }

    public static void mmul(float[][] a, float[][] b, float[][] c) {
        mmul(a, b, c, 0, a.length, 0, b.length, 0);
    }
    public static void mmul(float[][] a, float[][] b, float[][] c,
                            final int aRowStart, final int aRowEnd,
                            final int bRowStart, final int bRowEnd, int aOffset) {
        int aRange = aRowEnd - aRowStart;
        int bRange = bRowEnd - bRowStart;
        if (aRange * bRange > 1024) {
            int split;
            if (aRange < bRange) {
                split = (bRowEnd + bRowStart) / 2;
                mmul(a, b, c, aRowStart, aRowEnd, split, bRowEnd, aOffset);
                mmul(a, b, c, aRowStart, aRowEnd, bRowStart, split, aOffset);
            } else {
                split = (aRowEnd + aRowStart) / 2;
                mmul(a, b, c, split, aRowEnd, bRowStart, bRowEnd, aOffset);
                mmul(a, b, c, aRowStart, split, bRowStart, bRowEnd, aOffset);
            }
        } else {
            int i, k, j;
            for (i = aRowStart; i < aRowEnd; i++) {
                final float[] iRowA = a[i];
                final float[] iRowC = c[i - aOffset];
                for (k = bRowStart; k < bRowEnd; k++) {
                    final float[] kRowB = b[k];
                    final float ikA = iRowA[k];
                    for (j = 0; j < b[0].length; j++) {
                        iRowC[j] = iRowC[j] + ikA * kRowB[j];
                    }
                }
            }
        }
    }

}
