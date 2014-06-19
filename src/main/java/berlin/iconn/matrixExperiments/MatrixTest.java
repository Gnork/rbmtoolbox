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
public class MatrixTest {

    private static final Random RANDOM = new Random();
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

        while (n >= penalty) {
            FloatMatrix a = FloatMatrix.rand(m, n).add(0.0f).mul(1);
            FloatMatrix b = FloatMatrix.rand(n, m).add(0.5f).mul(2);

            float[][] x = a.toArray2();
            float[][] y = b.toArray2();

//            for (int i = 0; i < x.length; i++) {
//                for (int j = 0; j < x[0].length; j++) {
//                    x[i][j] = (float) RANDOM.nextGaussian();
//                }
//            }
//            for (int i = 0; i < y.length; i++) {
//                for (int j = 0; j < y[0].length; j++) {
//                    y[i][j] = (float) RANDOM.nextGaussian();
//                }
//            }

            float[][] c = new float[x.length][y[0].length];
            float[][] d = new float[x.length][y[0].length];

            System.err.println("m:" + m + " n: " + n);
            long start = System.currentTimeMillis();
            mmul(x, y, c);
            System.out.println("mmul: "
                    + (System.currentTimeMillis() - start) + "ms");
            start = System.currentTimeMillis();
            approx_mmul2(x, y, d, 0.2f);
            System.out.println("approx_mmul: "
                    + (System.currentTimeMillis() - start) + "ms");


            FloatMatrix r1 = new FloatMatrix(c);
            System.out.println(r1.getRow(0).getRange(0, 10));
            FloatMatrix r2 = new FloatMatrix(d);
            System.out.println(r2.getRow(0).getRange(0, 10));
            FloatMatrix diff = r1.sub(r2);
            double error = Math.sqrt(diff.mul(diff).sum() / (m * m));

            System.out.println("Error: " + error);

            m *= 2;
            n /= 2;
        }
    }

    public static void approx_mmul2(final float[][] a, final float[][] b,
                                    final float[][] c, float amount) {

        amount *= 0.7f;
        if(b.length < 10 || amount > 0.7f){
            mmul(a, b, c);
            return;
        }
        int pick =  (int) (b.length * amount);
        if(pick < 10) pick = 10;


        final float[][] aT = transpose(a);
        final float[] aprops = getLengths(aT);
        final float sumA = sum(aprops);
        final float[] bprops = getLengths(b);
        float sumB = sum(bprops);

        final float[][] cT = new float[pick][];
        final float[][] r = new float[pick][];
        int index = 0;
        while(index < pick) {
            final int rand = getIndex(aT.length);
            final float coeffA = (float) Math.sqrt(pick * aprops[rand] / sumA);
            final float coeffB = (float) Math.sqrt(pick * bprops[rand] / sumB);
            cT[index] = rescale(aT[rand], coeffA);
            r[index] = rescale(b[rand], coeffB);
            index++;
        }
        mmul(transpose(cT), r, c);
    }

    private static int getIndex(int size) {
        double sqrtSize  = (Math.sqrt(size));
        double x = (RANDOM.nextGaussian() * sqrtSize * 0.125 + sqrtSize * 0.5);
        double y = (RANDOM.nextGaussian() * sqrtSize * 0.125 + sqrtSize * 0.5);
        double index = y * sqrtSize + x;
        System.out.println((int)index);
        if(index >= size) return  size - 1;
        if(index < 0) return 0;
        return (int) index;
    }

    private static float[] rescale(float[] a, float scale) {

        float[] result = new float[a.length];
        for (int i = 0; i < result.length; i++) {
            result[i] = a[i] / scale;
        }
        return result;
    }
    private static void normalize(float[] lengths) {
        float sum = sum(lengths);

        for (int i = 0; i < lengths.length; i++) {
            lengths[i] /= sum;
        }

    }

    private static float sum(float[] a) {
        float result = 0.0f;
        for (int i = 0; i < a.length; i++) {
            result += a[i];
        }
        return result;
    }

    public static float[] getLengths(float[][] a) {
        float[] result = new float[a.length];
        for (int i = 0; i < result.length; i++) {
            result[i] = getLength(a[i]);
        }
        return result;
    }

    public static float getLength(float[] a) {
        float result = 0;
        for (int i = 0; i < a.length; i++) {
            result += a[i] * a[i];
        }
        return (float) Math.sqrt(result);
    }

    private static float[][] transpose(float[][] b) {
        float[][] result = new float[b[0].length][b.length];

        for (int i = 0; i < result.length; i++) {
            for (int j = 0; j < result[0].length; j++) {
                result[i][j] = b[j][i];
            }
        }

        return result;
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

    public static void mmul(float[][] a, float[][] b, float[][] c) {

        final int parallelThresHold = 0x40000;
        if (a.length * a[0].length < parallelThresHold
                || b.length * b[0].length < parallelThresHold) {
            MatrixMultiplyKernel.mmul(a, b, c);
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

