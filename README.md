# non-local-means--image-denoising

The main idea of this problem is to denoise images with Gaussian noise input, using the non-local-mean algorithm. The non-local means filtering takes a mean of all pixels in the image, weighted by how similar these pixels are to the target pixel.


# Σειριακή υλοποίηση

The denoise() function is called N * N times in main () where each time it accepts as an argument a pixel of the image with Gaussian noise and gradually through the weights w calculates and returns the corresponding pixel that will have the denoised image. 
-The "patch (..., int i, int j, ...)" function returns the square neighborhood centered on the "pixel (i, j)" of the image.

# Παράλληλη υλοποίηση με cuda
In the parallel implementation to accelerate the process are used the functions __global__ void CudaPatch() and __global__ void nonLocalMeans(). these two functions replace the for loops in main where the patch() and denoise() functions which became __devised__ functions. 
