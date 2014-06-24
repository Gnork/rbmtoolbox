extern "C"

__global__
void contrastiveDivergence(float* positive, float* negative, float* weights, float learningRate, int n) 
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
	{
		weights[i] = weights[i] + (positive[i] - negative[i]) * learningRate;
	}
}