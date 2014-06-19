package berlin.iconn.matrixExperiments;

/**
 * Created by Moritz on 6/14/2014.
 */
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.LinkedBlockingQueue;

public class Matrix {

    public static final float INV_255 = 1.0f / 255.0f;
    public static final float MUL_255 = 255.0f;
    private static final Random RANDOM = new Random();
    private float[][] values;

    public Matrix(float[][] values) {
        this.values = values;
    }


    public static Matrix createGaussianRandom(int rows, int columns, float mean, float range) {
        float[][] field = new float[rows][columns];
        for (int i = 0; i < rows; i++) {
            float[] row = field[i];
            int y = i * columns;
            for (int j = 0; j < columns; j++) {
                row[j] = mean + (float)RANDOM.nextGaussian() * range;
            }
        }
        return new Matrix(field);
    }
    public static Matrix createRandom(int rows, int columns) {
        float[][] field = new float[rows][columns];
        for (int i = 0; i < rows; i++) {
            float[] row = field[i];
            int y = i * columns;
            for (int j = 0; j < columns; j++) {
                row[j] = RANDOM.nextFloat() ;
            }
        }
        return new Matrix(field);
    }
    public Matrix plus(Matrix b) {
        float[][] other = b.toArray2();
        float[][] result = new float[other.length][other[0].length];
        for (int i = 0; i < other.length; i++) {
            for (int j = 0; j < other[0].length; j++) {
                result[i][j] = this.values[i][j] + other[i][j];
            }
        }
        return new Matrix(result);
    }

    public Matrix subtract(Matrix b) {
        float[][] other = b.toArray2();
        float[][] result = new float[other.length][other[0].length];
        for (int i = 0; i < other.length; i++) {
            for (int j = 0; j < other[0].length; j++) {
                result[i][j] = this.values[i][j] - other[i][j];
            }
        }
        return new Matrix(result);
    }

    public Matrix scale(float v) {
        float[][] result = new float[values.length][values[0].length];
        for (int i = 0; i < values.length; i++) {
            for (int j = 0; j < values[0].length; j++) {
                result[i][j] = this.values[i][j] * v;
            }
        }
        return new Matrix(result);
    }

    public Matrix subtract(float v) {
        float[][] result = new float[values.length][values[0].length];
        for (int i = 0; i < values.length; i++) {
            for (int j = 0; j < values[0].length; j++) {
                result[i][j] = this.values[i][j] - v;
            }
        }
        return new Matrix(result);
    }

    public Matrix plus(float v) {
        float[][] result = new float[values.length][values[0].length];
        for (int i = 0; i < values.length; i++) {
            for (int j = 0; j < values[0].length; j++) {
                result[i][j] = this.values[i][j] + v;
            }
        }
        return new Matrix(result);
    }

    public Matrix multiply(float v) {
        float[][] result = new float[values.length][values[0].length];
        for (int i = 0; i < values.length; i++) {
            for (int j = 0; j < values[0].length; j++) {
                result[i][j] = this.values[i][j] * v;
            }
        }
        return new Matrix(result);
    }

    public Matrix multiply(Matrix v) {
        float[][] result = new float[getRows()][getColumns()];
        float[][] other = v.toArray2();
        for (int i = 0; i < values.length; i++) {
            for (int j = 0; j < values[0].length; j++) {
                result[i][j] = this.values[i][j] * other[i][j];
            }
        }
        return new Matrix(result);
    }

    public Matrix multiplyWithRandom() {
        float[][] result = new float[getRows()][getColumns()];
        for (int i = 0; i < values.length; i++) {
            for (int j = 0; j < values[0].length; j++) {
                result[i][j] = this.values[i][j] * RANDOM.nextFloat();
            }
        }
        return new Matrix(result);
    }

    public Matrix matrixMultiply(Matrix b) {
        float[][] other = b.toArray2();
        float[][] result = new float[values.length][other[0].length];
        mmul(this.values, other, result);

        return new Matrix(result);
    }

    private static void mmul(float[][] a, float[][] b, float[][] c) {

        final int parallelThresHold = MatrixMultiplyKernel.parallelThresHold;;
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

    public Matrix transpose() {
        int cols = getColumns();
        int rows = getRows();
        float[][] result = new float[cols][rows];
        for (int i = 0; i < rows; i++) {
            float[] row = values[i];
            for (int j = 0; j < cols; j++) {
                result[j][i] = row[j];
            }
        }
        return new Matrix(result);
    }

    public static Matrix generateFromImagesYCbCr(BufferedImage[] images, int width, int height) {
        float[][] result = new float[images.length][width * height * 3 + 1];
        Matrix matrix = new Matrix(result);
        matrix.resetDataBias();

        for (int i = 0; i < images.length; i++) {
            int[] pixels = extractPixels(width, height, images[i]);

            for (int j = 0; j < pixels.length; j++) {
                int pixel = pixels[j];

                int red = (pixel >> 16) & 0xff;
                int green = (pixel >> 8) & 0xff;
                int blue = (pixel & 0xff);
                float y = (0.299f * red + 0.587f * green + 0.114f * blue) * INV_255;
                float cb = (-0.168736f * red + -0.331264f * green + 0.5f * blue + 128f) * INV_255;
                float cr = (0.5f * red + -0.418688f * green + -0.081312f * blue + 128f) * INV_255;

                result[i][j + 1] = y;
                result[i][j + 1 + pixels.length] = cb;
                result[i][j + 1 + 2 * pixels.length] = cr;
            }
        }
        return matrix;
    }

    public static Matrix generateFromImagesGrayScale(BufferedImage[] images, int width, int height) {
        float[][] result = new float[images.length][width * height + 1];
        Matrix matrix = new Matrix(result);
        matrix.resetDataBias();

        for (int i = 0; i < images.length; i++) {
            int[] pixels = extractPixels(width, height, images[i]);

            for (int j = 0; j < pixels.length; j++) {
                int pixel = pixels[j];

                int red = (pixel >> 16) & 0xff;
                int green = (pixel >> 8) & 0xff;
                int blue = (pixel & 0xff);
                float y = (0.299f * red + 0.587f * green + 0.114f * blue) * INV_255;

                result[i][j + 1] = y;
            }
        }
        return matrix;
    }

    public void resetDataBias() {
        for (int i = 0; i < values.length; i++) {
            values[i][0] = 1.0f;
        }
    }

    public BufferedImage[] toImages(int width, int height) {
        if (width * height != getColumns()) {
            throw new IllegalArgumentException("Dimension does not fit matrix' column size");
        }
        BufferedImage[] result = new BufferedImage[getRows()];
        for (int i = 0; i < result.length; i++) {
            BufferedImage bufferedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
            int[] pixels = new int[getColumns()];
            for (int j = 0; j < pixels.length; j++) {
                int r = (int) (values[i][j + 1] * MUL_255) & 0xff;
                int g = (int) (values[i][j + 1 + pixels.length] * MUL_255) & 0xff;
                int b = (int) (values[i][j + 1 + 2 * pixels.length] * MUL_255) & 0xff;
                pixels[j] = (r << 16) | (g << 8) | b;
            }
            bufferedImage.setRGB(0, 0, width, height, pixels, 0, width);
        }
        return result;
    }

    public static Matrix generateFromImagesRGB(BufferedImage[] images, int width, int height) {
        float[][] result = new float[images.length][width * height * 3 + 1];
        Matrix matrix = new Matrix(result);
        matrix.resetDataBias();

        for (int i = 0; i < images.length; i++) {
            int[] pixels = extractPixels(width, height, images[i]);

            for (int j = 0; j < pixels.length; j++) {
                int pixel = pixels[j];

                result[i][j + 1] = ((pixel >> 16) & 0xff) * INV_255;
                result[i][j + 1 + pixels.length] = ((pixel >> 8) & 0xff) * INV_255;
                result[i][j + 1 + 2 * pixels.length] = (pixel & 0xff) * INV_255;
            }
        }
        return matrix;
    }

    private static int[] extractPixels(int width, int height, BufferedImage image) {
        BufferedImage newImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = (Graphics2D) newImage.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                RenderingHints.VALUE_INTERPOLATION_BICUBIC);
        g.drawImage(image, 0, 0, width, height, null);
        g.dispose();
        int[] pixels = new int[height * width];
        newImage.getRGB(0, 0, width, height, pixels, 0, width);
        return pixels;
    }

    public static Matrix generateWeights(int input, int output, float scale, long seed) {
        RANDOM.setSeed(seed);
        return generateGaussianRandomWeights(input, output, scale);
    }

    private static Matrix generateGaussianRandomWeights(int input, int output, float scale) {
        float[][] result = new float[input + 1][output + 1];
        for (int i = 1; i < result.length; i++) {
            float[] row = result[i];
            for (int j = 1; j < row.length; j++) {
                row[j] = (float) (RANDOM.nextGaussian() * scale);
            }
        }
        return new Matrix(result);
    }

    public static Matrix generateWeights(int input, int output, float scale) {
        return generateGaussianRandomWeights(input, output, scale);
    }

    public float[][] toArray2() {
        return values;
    }

    public int getRows() {
        return values.length;
    }

    public int getColumns() {
        return values[0].length;
    }

    public void iApplyLogistic() {

        for (int i = 0; i < values.length; i++) {
            float[] fs = values[i];
            for (int j = 0; j < fs.length; j++) {
                fs[j] = (float)(1.0 / ( Math.exp(-fs[j]) + 1.0));
            }
        }
    }

    public float getMeanSquareErrorWithoutBias(Matrix b) {
        float sum = squareSum(b);
        return (float) Math.sqrt(sum / ((values[0].length - 1) * (values.length)));
    }

    public float getMeanSquareError(Matrix b) {
        float sum = squareSum(b);
        return (float) Math.sqrt(sum / ((values[0].length) * (values.length)));
    }

    private float squareSum(Matrix b) {
        float[][] other = b.values;
        float sum = 0;
        float diff = 0;
        for (int i = 0; i < other.length; i++) {
            float[] bRow = other[i];
            float[] aRow = values[i];
            for (int j = 0; j < bRow.length; j++) {
                diff = aRow[j] - bRow[j];
                sum = diff * diff;
            }
        }
        return sum;
    }


    public void print() {
        for (int i = 0; i < values.length; i++) {
            for (int j = 0; j < values[0].length; j++) {
                System.out.print(String.format("%.5f",values[i][j]) + ", ");
            }
            System.out.println();
        }
        System.out.println();
    }
}

