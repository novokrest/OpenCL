__kernel void gpu_convolution_gmem(__global float * input, __global float * mask, 
                                   __global float * output, int n, int m)
{
   int idx = get_global_id(0);
   int row = idx / n;
   int column = idx % n;

   if (row >= n || column >= n)
   {
	   return;
   }

   float res = 0;
   for (int i = 0; i < m; ++i)
   {
	   for (int j = 0; j < m; ++j)
	   {
		   int input_row = row - m / 2 + i;
		   int input_column = column - m / 2 + j;
		   if (input_row >= 0 && input_row < n && input_column >= 0 && input_column < n)
		   {
			   res += mask[i * m + j] * input[input_row * n + input_column];
		   }
	   }
   }

   output[idx] = res;
}
