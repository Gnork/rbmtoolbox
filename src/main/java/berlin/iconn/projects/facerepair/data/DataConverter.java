package berlin.iconn.projects.facerepair.data;

import java.awt.image.BufferedImage;

/**
 * Created by Moritz on 5/18/2014.
 */
public class DataConverter {

    public static float[] processPixelData(BufferedImage image, int edgeLength, boolean binarize, boolean invert, float minData, float maxData, boolean isRgb) {
        float[] result;
        if(isRgb) {
            result = processPixelRGBData(image, edgeLength, binarize, invert, minData, maxData);
        } else {
            result = processPixelIntensityData(image, edgeLength, binarize, invert, minData, maxData);
        }
        return result;
    }

    public static float[] processPixelData(float[] imageData, int edgeLength, boolean binarize, boolean invert, float minData, float maxData, boolean isRgb) {
        return processPixelIntensityData(imageData, edgeLength, binarize, invert, minData, maxData);
    }

    public static BufferedImage pixelDataToImage(float[] data, float minData, boolean isRgb) {
        if(isRgb) {
            return pixelRGBDataToImage(data, minData);
        } else {
            return pixelIntensityDataToImage(data, minData);
        }
    }

    private static float[] processPixelIntensityData(float[] imageData, int edgeLength, boolean binarize, boolean invert, float minData, float maxData) {
        float[] data = new float[imageData.length];

        for (int i = 0; i < imageData.length; i++) {

            float intensity = imageData[i];

            if(invert) {
                intensity = 1.0f - intensity;
            }
            data[i] = intensity;
        }

        if(binarize) {
            binarizeImage(data);
        }

        float scale = maxData - minData;
        for(int i = 0; i < data.length; i++) {
            data[i] = minData + data[i] * scale;
        }

        return data;
    }

    private static float[] processPixelIntensityData(BufferedImage image, int edgeLength, boolean binarize, boolean invert, float minData, float maxData) {
        float[] data = new float[edgeLength * edgeLength];

        ImageScaler imageScaler = new ImageScaler(image);
        BufferedImage scaledImage = imageScaler.scale(edgeLength);
        int[] pixels = scaledImage.getRGB(0, 0, edgeLength, edgeLength, null, 0, edgeLength);

        for (int p = 0; p < pixels.length; p++) {
            int argb = pixels[p];

            int r = (argb >> 16) & 0xFF;
            int g = (argb >> 8) & 0xFF;
            int b = (argb) & 0xFF;

            float intensity = Math.max(0.0f, Math.min(1.0f, (float) (0.299 * r + 0.587 * g + 0.114 * b) / 255.0f));
            data[p] = intensity;
        }

        return processPixelIntensityData(data, edgeLength, binarize, invert, minData, maxData);
    }

    private static float[] processPixelRGBData(BufferedImage image, int edgeLength, boolean binarize, boolean invert, float minData, float maxData) {
        float[] data = new float[edgeLength * edgeLength * 3];

        ImageScaler imageScaler = new ImageScaler(image);
        BufferedImage scaledImage = imageScaler.scale(edgeLength);
        int[] pixels = scaledImage.getRGB(0, 0, edgeLength, edgeLength, null, 0, edgeLength);

        for (int p = 0; p < pixels.length; p++) {
            int argb = pixels[p];

            float r = ((argb >> 16) & 0xFF) / 255.0f;
            float g = ((argb >> 8) & 0xFF) / 255.0f;
            float b = ((argb) & 0xFF) / 255.0f;

            int pixel = p * 3;

            data[pixel] = r;
            data[pixel + 1] = g;
            data[pixel + 2] = b;
        }

        return processPixelIntensityData(data, edgeLength, binarize, invert, minData, maxData);
    }

    private static BufferedImage pixelRGBDataToImage(float[] data, float minData) {
        int edgeLength = (int) Math.sqrt(data.length / 3);
        BufferedImage image = new BufferedImage(edgeLength, edgeLength, BufferedImage.TYPE_INT_RGB);

        int[] rgb = new int[data.length / 3];
        for (int i = 0; i < rgb.length; i++) {

            int pixel = i * 3;

            float rShiftet = (data[pixel] + Math.abs(minData));
            float gShiftet = (data[pixel + 1] + Math.abs(minData));
            float bShiftet = (data[pixel + 2] + Math.abs(minData));

            int r = (int)(rShiftet * 255);
            int g = (int)(gShiftet * 255);
            int b = (int)(bShiftet * 255);

            rgb[i] = (0xFF << 24) | (r << 16) | (g << 8) | b;
        }

        image.setRGB(0, 0, edgeLength, edgeLength, rgb, 0, edgeLength);

        return image;
    }

    private static BufferedImage pixelIntensityDataToImage(float[] data, float minData) {
        int edgeLength = (int) Math.sqrt(data.length);
        BufferedImage image = new BufferedImage(edgeLength, edgeLength, BufferedImage.TYPE_INT_RGB);

        int[] rgb = new int[data.length];
        for (int i = 0; i < rgb.length; i++) {
            int value = (int)(data[i] * 255);
            rgb[i] = (0xFF << 24) | (value << 16) | (value << 8) | value;

        }

        image.setRGB(0, 0, edgeLength, edgeLength, rgb, 0, edgeLength);

        return image;
    }

    private static BufferedImage pixelIntensityDataToImageConvolution(float[] data, float minData) {
        int edgeLength = (int) Math.sqrt(data.length);
        BufferedImage image = new BufferedImage(edgeLength, edgeLength, BufferedImage.TYPE_INT_RGB);

        int[] rgb = new int[data.length];
        for (int i = 0; i < rgb.length; i++) {
            rgb[i] = (0xFF << 24) | (255 << 16) | (255 << 8) | 255;
        }

        for(int y = 1; y < edgeLength - 1; y++) {
            for(int x = 1; x < edgeLength - 1; x++) {
                int pos = y * edgeLength + x;
                float dataShiftet = (data[pos] + Math.abs(minData));
                if(dataShiftet < 0) {
                    rgb[pos] = 0xFFFF0000;
                } else if(dataShiftet > 1) {
                    rgb[pos] = 0xFF00FF00;
                } else {
                    int value = (int)(dataShiftet * 255);
                    rgb[pos] = (0xFF << 24) | (value << 16) | (value << 8) | value;
                }
            }
        }



        image.setRGB(0, 0, edgeLength, edgeLength, rgb, 0, edgeLength);

        return image;
    }

    private static void binarizeImage(float[] data) {
        float threshold = findOptimalThreshold(data);
        for(int i = 0; i < data.length; i++) {
            float value = data[i];
            data[i] = value < threshold ? 0.0f : 1.0f;
        }
    }

    private static float findOptimalThreshold(float pixels[]) {

        float[] hist = new float[256];

        for(int i = 0; i < pixels.length; i++) {
            int gray = (int)(Math.round(pixels[i] * 255));
            hist[gray] += 1.0f/pixels.length;
        }

        int median = 0;
        float medianValue = 0;
        for(int i = 0; i < hist.length; i++) {
            medianValue += hist[i];
            if(medianValue >= 0.5) {
                median = i;
                break;
            }
        }

        int t = (median == 0) ? 128 : median;
        int t_last = 0;

        while (t != t_last) {

            t_last = t;

            float[] hist1 = new float[t];
            float[] hist2 = new float[256 - t];

            System.arraycopy(hist, 0, hist1, 0, hist1.length);
            System.arraycopy(hist, hist1.length, hist2, 0, hist2.length);

            float u1 = isoData(hist1, 0);
            float u2 = isoData(hist2, t);

            t = (int) ((u1 + u2) / 2);
        }

        return t / 255.0f;
    }

    private static float isoData(float[] hist, int offset) {
        float P = 0;

        for(int i = 0; i < hist.length; i++) {
            P += hist[i];
        }

        float u = 0;

        for(int i = 0; i < hist.length; i++) {
            u += hist[i] * (i + offset);
        }
        u /= P;

        return u;
    }

    public static float[][] dataSetToArray(DataSet[] dataSet) {
        float[][] result = new float[dataSet.length][];
        for (int i = 0; i < dataSet.length; ++i) {
            result[i] = dataSet[i].getData();
        }
        return result;
    }

    public static float[] pad(float[] data, int dataEdgeLength, int padding) {
        int newEdgeLength = dataEdgeLength + padding * 2;
        float[] result = new float[newEdgeLength * newEdgeLength];

        for (int y = 0; y < newEdgeLength; y++) {
            for (int x = 0; x < newEdgeLength; x++) {

                int pos = y * newEdgeLength + x;
                if (y < padding || x < padding || y >= dataEdgeLength + padding || x >= dataEdgeLength + padding) {
                    result[pos] = 0.0f;
                } else {
                    int posm = (y - padding) * (newEdgeLength - padding * 2) + x - padding;
                    result[pos] = data[posm];
                }

            }
        }
        return result;
    }
    
    public static float[] processPixelL1RGB0Data(BufferedImage image) {
        float[] data = new float[image.getWidth() * image.getHeight() * 4];

        int[] pixels = image.getRGB(0, 0, image.getWidth(), image.getHeight(), null, 0, image.getWidth());

        for (int p = 0; p < pixels.length; p++) {
            int argb = pixels[p];
            
            LAB lab = LAB.fromRGB(((argb >> 16) & 0xFF), ((argb >> 8) & 0xFF), (argb & 0xFF), 0);
            float L = (float) Math.max(0, Math.min(1, lab.L / 100d));
            
            int offset = p * 4;

            data[offset] = L;
            data[offset + 1] = 0;
            data[offset + 2] = 0;
            data[offset + 3] = 0;
        }

        return data;
    }
    
    public static float[] processPixelLRGBData(BufferedImage image) {
        float[] data = new float[image.getWidth() * image.getHeight() * 4];

        int[] pixels = image.getRGB(0, 0, image.getWidth(), image.getHeight(), null, 0, image.getWidth());

        for (int p = 0; p < pixels.length; p++) {
            int argb = pixels[p];
            
            LAB lab = LAB.fromRGB(((argb >> 16) & 0xFF), ((argb >> 8) & 0xFF), (argb & 0xFF), 0);
            float L = (float) Math.max(0, Math.min(1, lab.L / 100d));
            float r = ((argb >> 16) & 0xFF) / 255f;
            float g = ((argb >> 8) & 0xFF) / 255f;
            float b = (argb & 0xFF) / 255f;
            
            int offset = p * 4;

            data[offset] = L;
            data[offset + 1] = r;
            data[offset + 2] = g;
            data[offset + 3] = b;
        }

        return data;
    }
    
    public static BufferedImage pixelLRGBDataToImage(float[] data, int width, int height) {
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

        int[] rgb = new int[data.length / 4];

        for (int p = 0; p < rgb.length; p++) {
            int pixel = p * 4;

            int r = (int) (data[pixel + 1] * 255);
            int g = (int) (data[pixel + 2] * 255);
            int b = (int) (data[pixel + 3] * 255);

            rgb[p] = (0xFF0000 & (r << 16)) | (0x00FF00 & (g << 8)) | (0xFF & b);
        }

        image.setRGB(0, 0, width, height, rgb, 0, width);

        return image;
    }
    
    public static float[] processPixelRGBData(BufferedImage image) {
        float[] data = new float[image.getWidth() * image.getHeight() * 3];

        int[] pixels = image.getRGB(0, 0, image.getWidth(), image.getHeight(), null, 0, image.getWidth());

        for (int p = 0; p < pixels.length; p++) {
            int argb = pixels[p];

            float r = ((argb >> 16) & 0xFF) / 255f;
            float g = ((argb >> 8) & 0xFF) / 255f;
            float b = (argb & 0xFF) / 255f;
            
            int offset = p * 3;

            data[offset] = r;
            data[offset + 1] = g;
            data[offset + 2] = b;
        }

        return data;
    }
    
    public static BufferedImage pixelRGBDataToImage(float[] data, int width, int height) {
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

        int[] rgb = new int[data.length / 3];

        for (int p = 0; p < rgb.length; p++) {
            int pixel = p * 3;

            int r = (int) (data[pixel] * 255);
            int g = (int) (data[pixel + 1] * 255);
            int b = (int) (data[pixel + 2] * 255);

            rgb[p] = (0xFF0000 & (r << 16)) | (0x00FF00 & (g << 8)) | (0xFF & b);
        }

        image.setRGB(0, 0, width, height, rgb, 0, width);

        return image;
    }

    public static float[] processPixelLABData(BufferedImage image) {
        float[] data = new float[image.getWidth() * image.getHeight() * 3];

        int[] pixels = image.getRGB(0, 0, image.getWidth(), image.getHeight(), null, 0, image.getWidth());

        for (int p = 0; p < pixels.length; p++) {
            int argb = pixels[p];

            LAB lab = LAB.fromRGB(((argb >> 16) & 0xFF), ((argb >> 8) & 0xFF), (argb & 0xFF), 0);

            float L = (float) Math.max(0, Math.min(1, lab.L / 100d));
            float a = (float) Math.max(0, Math.min(1, (lab.a + 127.5d) / 255d));
            float b = (float) Math.max(0, Math.min(1, (lab.b + 127.5d) / 255d));
            
            int offset = p * 3;

            data[offset] = L;
            data[offset + 1] = a;
            data[offset + 2] = b;
        }

        return data;
    }

    public static BufferedImage pixelLABDataToImage(float[] data, int width, int height) {
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

        int[] rgb = new int[data.length / 3];

        for (int p = 0; p < rgb.length; p++) {
            int pixel = p * 3;

            int L = (int) (data[pixel] * 100);
            int a = (int) (data[pixel + 1] * 255 - 127.5);
            int b = (int) (data[pixel + 2] * 255 - 127.5);

            rgb[p] = new LAB(L, a, b).rgb();
        }

        image.setRGB(0, 0, width, height, rgb, 0, width);

        return image;
    }
}
