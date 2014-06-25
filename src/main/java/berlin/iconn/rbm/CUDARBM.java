package berlin.iconn.rbm;
import static jcuda.driver.JCudaDriver.*;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;

import org.jblas.FloatMatrix;

import berlin.iconn.rbm.dataprovider.ATrainingDataProvider;
import berlin.iconn.rbm.learningRate.ILearningRate;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.jcublas.JCublas;


public class CudaRBM extends RBM {


    private static CUdevice device;
    private static CUcontext context;
    private static CUfunction sigmoid;
    private static CUfunction contrastiveDivergence;

    private static int blockDim = 256;
    static {
        initCudaRBM();
        sigmoid = loadFunction("sigmoid.cu", "sigmoid");
        contrastiveDivergence = loadFunction("contrastive_divergence.cu", "contrastiveDivergence");
    }

    private int miniBatchCountOnCudaDevice;

    public CudaRBM(FloatMatrix weights) {
        super(weights);
    }

    @Override
    public void train(ATrainingDataProvider dataProvider,
                      StoppingCondition stop, ILearningRate learningRate) {

        FloatMatrix data = dataProvider.getDataWithBias();
        int weightColumns = super.weights.getColumns();
        int visibleColumns = data.getColumns();
        int visibleRows = data.getRows();

        int visibleSize =  visibleColumns * visibleRows;
        int hiddenSize = visibleRows * super.weights.getColumns();
        int weightsSize = weightColumns * super.weights.getRows();

        Pointer weights = new Pointer();
        JCublas.cublasAlloc(weightsSize, Sizeof.FLOAT, weights);
        JCublas.cublasSetVector(weightsSize, Sizeof.FLOAT, Pointer.to(super.weights.data), 1, weights, 1);

        Pointer visible = new Pointer();
        JCublas.cublasAlloc(visibleSize, Sizeof.FLOAT, visible);

        Pointer hidden = new Pointer();
        JCublas.cublasAlloc(hiddenSize, Sizeof.FLOAT, hidden);

        Pointer positive = new Pointer();
        JCublas.cublasAlloc(weightsSize, Sizeof.FLOAT, positive);

        Pointer negative = new Pointer();
        JCublas.cublasAlloc(weightsSize, Sizeof.FLOAT, negative);

        while(stop.isNotDone()) {
            System.out.println("next epoch");
            float[] miniBatch = dataProvider.getDataWithBias().data;
            JCublas.cublasSetVector(visibleSize, Sizeof.FLOAT, Pointer.to(miniBatch), 1, visible, 1);
            updateWeights(
                    visible, weights,
                    hidden, positive, negative,
                    visibleRows, visibleColumns, weightColumns,
                    learningRate.getRate());
            stop.update(0.0f);
            dataProvider.changeDataAtTraining();
        }

        JCublas.cublasGetVector(weightsSize, Sizeof.FLOAT, weights, 1, Pointer.to(super.weights.data), 1);
        JCublas.cublasFree(weights);
        JCublas.cublasFree(positive);
        JCublas.cublasFree(negative);
        JCublas.cublasFree(hidden);
    }

    private void updateWeights(Pointer visible, Pointer weights, Pointer hidden, Pointer positive, Pointer negative,
                               int visibleRows, int visibleColumns, int weightColumns, float learningRate) {

        int hiddenSize = visibleRows * weightColumns;
        mmul(visible, visibleRows, visibleColumns, weights, weightColumns, hidden);
        applyLogistic(hidden, hiddenSize);
//		show(hidden, visibleRows, weightColumns, "hidden");

        mmulTransposeA(visible, visibleRows, visibleColumns, hidden, weightColumns, positive);
//		show(positive, visibleColumns, weightColumns, "postive");

        mmulTransposeB(hidden, visibleRows, weightColumns, weights, visibleColumns, visible);
        applyLogistic(visible, visibleRows * visibleColumns);
        resetBias(visible, visibleRows, visibleColumns);
//		show(visible, visibleRows, visibleColumns, "visible");			

        mmul(visible, visibleRows, visibleColumns, weights, weightColumns, hidden);
        applyLogistic(hidden, hiddenSize);
//		show(hidden, visibleRows, weightColumns, "hidden 2");

        mmulTransposeA(visible, visibleRows, visibleColumns, hidden, weightColumns, negative);
//		show(negative, visibleColumns, weightColumns, "negative");

        contrastiveDivergence(positive, negative, weights, visibleColumns * weightColumns, learningRate / visibleRows);
//		show(weights, visibleColumns, weightColumns, "weights 2");			
    }


    private void mmul(Pointer a, int aRows, int aColumnsbRows,
                      Pointer b, int bColumns,
                      Pointer c) {
        JCublas.cublasSgemm('n', 'n',
                aRows, bColumns, aColumnsbRows,
                1.0f, a, aRows,
                b, aColumnsbRows,
                0.0f, c, aRows);
        cuCtxSynchronize();
    }

    private void mmulTransposeA(Pointer a, int aRowsbRows, int aColumns,
                                Pointer b, int bColumns,
                                Pointer c) {
        JCublas.cublasSgemm('t', 'n',
                aColumns, bColumns, aRowsbRows,
                1.0f, a, aRowsbRows,
                b, aRowsbRows,
                0.0f, c, aColumns);
        cuCtxSynchronize();
    }

    private void mmulTransposeB(Pointer a, int aRows, int aColumnsbColumns,
                                Pointer b, int bRows,
                                Pointer c) {
        JCublas.cublasSgemm('n', 't',
                aRows, bRows, aColumnsbColumns,
                1.0f, a, aRows,
                b, bRows,
                0.0f, c, aRows);
        cuCtxSynchronize();
    }


    public void resetBias(Pointer data, int rows, int columns) {
        float[] toReset = new float[rows];
        Arrays.fill(toReset, 1.0f);
        JCublas.cublasSetVector(rows,
                Sizeof.FLOAT, Pointer.to(toReset), 1, data, 1);

        cuCtxSynchronize();
    }

    public void applyLogistic(Pointer data, int length) {
        Pointer kernelParameters = Pointer.to(Pointer.to(data), Pointer.to(new int[]{length}));
        cuLaunchKernel(sigmoid,
                (int) Math.ceil(length / (double) blockDim), 1, 1,
                blockDim, 1, 1,
                0,
                null,
                kernelParameters, null);
        cuCtxSynchronize();
    }

    private void contrastiveDivergence(Pointer positive,
                                       Pointer negative,
                                       Pointer weights,
                                       int length,
                                       float learningRate) {
        Pointer kernelParameters = Pointer.to(
                Pointer.to(positive),
                Pointer.to(negative),
                Pointer.to(weights),
                Pointer.to(new float[]{learningRate}),
                Pointer.to(new int[]{length}));
        cuLaunchKernel(contrastiveDivergence,
                (int) Math.ceil(length / (double) blockDim), 1, 1,
                blockDim, 1, 1,
                0,
                null,
                kernelParameters, null);

        cuCtxSynchronize();
    }

    private static void show(Pointer pointer, int rows, int cols, String name) {
        FloatMatrix matrix = FloatMatrix.zeros(rows, cols);
        JCublas.cublasGetVector(rows * cols, Sizeof.FLOAT, pointer, 1, Pointer.to(matrix.data), 1);
        Main.print(matrix.toArray2(), name);
    }

    private static void initCudaRBM() {
        cuInit(0);
        JCublas.initialize();
        context = new CUcontext();
        device = new CUdevice();
        cuDeviceGet(device, 0);
        cuCtxCreate(context, 0, device);
    }

    private static CUfunction loadFunction(String cuFilePath, String name) {

        try {
            String ptxFileName = compilePtxFile(cuFilePath);

            CUmodule module = new CUmodule();
            cuModuleLoad(module, ptxFileName);

            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, module, name);

            return function;

        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }


    private static String compilePtxFile(String cuFileName) throws IOException
    {
        int endIndex = cuFileName.lastIndexOf('.');
        if (endIndex == -1)
        {
            endIndex = cuFileName.length()-1;
        }
        String ptxFileName = cuFileName.substring(0, endIndex+1)+"ptx";
        File ptxFile = new File(ptxFileName);
        if (ptxFile.exists())
        {
            return ptxFileName;
        }

        File cuFile = new File(cuFileName);
        if (!cuFile.exists())
        {
            throw new IOException("Input file not found: " + cuFileName);
        }
        String modelString = "-m" + System.getProperty("sun.arch.data.model");
        String command =
                "nvcc " + modelString + " -ptx "+
                        cuFile.getPath()+" -o "+ptxFileName;

        System.out.println("Executing\n"+command);
        Process process = Runtime.getRuntime().exec(command);

        String errorMessage =
                new String(toByteArray(process.getErrorStream()));
        String outputMessage =
                new String(toByteArray(process.getInputStream()));
        int exitValue = 0;
        try
        {
            exitValue = process.waitFor();
        }
        catch (InterruptedException e)
        {
            Thread.currentThread().interrupt();
            throw new IOException(
                    "Interrupted while waiting for nvcc output", e);
        }

        if (exitValue != 0)
        {
            System.out.println("nvcc process exitValue "+exitValue);
            System.out.println("errorMessage:\n"+errorMessage);
            System.out.println("outputMessage:\n"+outputMessage);
            throw new IOException(
                    "Could not create .ptx file: "+errorMessage);
        }

        System.out.println("Finished creating PTX file");
        return ptxFileName;
    }

    private static byte[] toByteArray(InputStream inputStream)
            throws IOException
    {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        byte buffer[] = new byte[8192];
        while (true)
        {
            int read = inputStream.read(buffer);
            if (read == -1)
            {
                break;
            }
            baos.write(buffer, 0, read);
        }
        return baos.toByteArray();
    }

}
