package berlin.iconn.matrixExperiments;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

import java.util.Random;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.LinkedBlockingQueue;

import org.jblas.FloatMatrix;

/**
 *
 * @author Moritz
 */
public class JBlasTest {

    /**
     * @param args
     *            the command line arguments
     */
    public static void main(String[] args) {
        matrixTest();

    }

    public static void matrixTest() {

        int m = 2048;
        int n = 4096;

        int penalty = m;

        Random random = new Random();
        while (n >= penalty) {
            FloatMatrix a = FloatMatrix.rand(m, n).sub(0.0f).mul(1);
            FloatMatrix b = FloatMatrix.rand(n, m).sub(0.0f).mul(1);

            float[][] x = a.toArray2();
            float[][] y = b.toArray2();

            for (int i = 0; i < x.length; i++) {
                for (int j = 0; j < x[0].length; j++) {
                    x[i][j] = (float) random.nextGaussian() + 1;
                }
            }
            for (int i = 0; i < y.length; i++) {
                for (int j = 0; j < y[0].length; j++) {
                    y[i][j] = (float) random.nextGaussian() + 1;
                }
            }

            float[][] c = new float[x.length][y[0].length];
            float[][] d = new float[x.length][y[0].length];

            System.err.println("m:" + m + " n: " + n);
            long start = System.currentTimeMillis();
            fast_mmul(x, y, c);
            System.out.println("fast_mmul: "
                    + (System.currentTimeMillis() - start) + "ms");
            start = System.currentTimeMillis();
            approx_mmul(x, y, d);
            System.out.println("approx_mmul: "
                    + (System.currentTimeMillis() - start) + "ms");
            // start = System.currentTimeMillis();
            // mmul(x, y, d);
            // System.out.println("mmul: " + (System.currentTimeMillis() -
            // start) + "ms");
            // start = System.currentTimeMillis();
            // a.mmul(b);
            // System.out.println("jblas_mmul: " + (System.currentTimeMillis() -
            // start) + "ms");

            FloatMatrix r1 = new FloatMatrix(c);
            System.out.println(r1.getRow(0));
            FloatMatrix r2 = new FloatMatrix(d);
            System.out.println(r2.getRow(0));
            FloatMatrix diff = r1.sub(r2);
            double error = Math.sqrt(diff.mul(diff).sum() / (m * m));

            System.out.println("Parallel Variance: " + error);

            m *= 2;
            n /= 2;
        }
    }

    public static void approx_mmul(final float[][] a, final float[][] b,
                                   final float[][] c) {

        float[][] aApprox = new float[a.length][2];
        float[][] bApprox = new float[2][b[0].length];

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                float aValue = a[i][j];
                float aValue2 = aValue * aValue;
                aApprox[i][0] += aValue;
                aApprox[i][1] += aValue2;
            }
        }

        for (int i = 0; i < b[0].length; i++) {
            for (int j = 0; j < b.length; j++) {
                float bValue = b[j][i];
                float bValue2 = bValue * bValue;
                bApprox[0][i] += bValue;
                bApprox[1][i] += bValue2;
            }
        }

        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b[0].length; j++) {
                c[i][j] = coefficients(aApprox[i][0], aApprox[i][1],
                        bApprox[0][j], bApprox[1][j], b.length);
                // c[i][j] = (float)Math.sqrt(-0.0625 * aApprox[i][0] *
                // bApprox[0][j] + 0.703125 * aApprox[i][1] * bApprox[1][j]);
            }
        }
    }

    private static float coefficients(float sumA, float sqrA, float sumB,
                                      float sqrB, int length) {
        length *= 2;

        float mean = (sumA + sumA) / length;
        float mean2 = mean * mean;
        float mean4 = mean2 * mean2;
        float variance = (sqrA + sqrB) / length - mean2;
        float variance2 = variance * variance;

        float b1 = 2 * mean2 * variance2 / (mean4 + variance2);
        float b2 = mean4 * (mean2 - variance)
                / ((mean2 + variance) * (mean4 + variance2));

        float result = b1 * sumA * sumB + b2 * sqrA * sqrB;

        return (float) ((result > 0) ? Math.sqrt(result) : -Math.sqrt(-result));

    }

    public static void print(float[][] data) {
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[0].length; j++) {
                System.out.print(Math.round(data[i][j]) + ", ");
            }
            System.out.println();
        }
        System.out.println();
    }

    public static void mmul(float[][] a, float[][] b, float[][] c,
                            final int aRowStart, final int aRowEnd, final int bRowStart,
                            final int bRowEnd) {
        int aRange = aRowEnd - aRowStart;
        int bRange = bRowEnd - bRowStart;
        if (aRange * bRange > 1024) {
            int split;
            if (aRange < bRange) {
                split = (bRowEnd + bRowStart) / 2;
                mmul(a, b, c, aRowStart, aRowEnd, split, bRowEnd);
                mmul(a, b, c, aRowStart, aRowEnd, bRowStart, split);
            } else {
                split = (aRowEnd + aRowStart) / 2;
                mmul(a, b, c, split, aRowEnd, bRowStart, bRowEnd);
                mmul(a, b, c, aRowStart, split, bRowStart, bRowEnd);
            }
        } else {
            int i, k, j;
            for (i = aRowStart; i < aRowEnd; i++) {
                final float[] iRowA = a[i];
                final float[] iRowC = c[i];
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

    public static void mmulp(float[][] a, float[][] b, float[][] c,
                             final int aRowStart, final int aRowEnd, final int bRowStart,
                             final int bRowEnd, int aOffset) {
        int aRange = aRowEnd - aRowStart;
        int bRange = bRowEnd - bRowStart;
        if (aRange * bRange > 1024) {
            int split;
            if (aRange < bRange) {
                split = (bRowEnd + bRowStart) / 2;
                mmulp(a, b, c, aRowStart, aRowEnd, split, bRowEnd, aOffset);
                mmulp(a, b, c, aRowStart, aRowEnd, bRowStart, split, aOffset);
            } else {
                split = (aRowEnd + aRowStart) / 2;
                mmulp(a, b, c, split, aRowEnd, bRowStart, bRowEnd, aOffset);
                mmulp(a, b, c, aRowStart, split, bRowStart, bRowEnd, aOffset);
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

    public static void mmul(final float[][] a, final float[][] b,
                            final float[][] c) {
        mmul(a, b, c, 0, a.length, 0, b.length);
    }

    public static void fast_mmul(float[][] a, float[][] b, float[][] c) {

        final int parallelThresHold = 0x40000;
        if (a.length * a[0].length < parallelThresHold
                || b.length * b[0].length < parallelThresHold) {
            mmul(a, b, c);
        } else {
            final LinkedBlockingQueue<OffsetArray> queue = new LinkedBlockingQueue<>();

            ForkJoinPool.commonPool().invoke(
                    new MatrixMultiplyKernel(queue, a, b));

            queue.forEach((array) -> {
                int offset = array.offset;
                float[][] values = array.array;
                for (int i = 0; i < values.length; i++) {
                    for (int j = 0; j < values[0].length; j++) {
                        c[i + offset][j] += values[i][j];
                    }

                }
            });
        }
    }

}

