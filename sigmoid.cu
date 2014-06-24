extern "C"
__global__
void sigmoid(float* a, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
	{
		a[i] = 1.0f / (expf(-a[i]) + 1.0f);
	}
}