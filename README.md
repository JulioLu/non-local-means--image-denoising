# non-local-means--image-denoising

The main idea of this problem is to denoise images with Gaussian noise input, using the non-local-mean algorithm. The non-local means filtering takes a mean of all pixels in the image, weighted by how similar these pixels are to the target pixel.

# Serial implementation
The denoise() function is called N * N times in main () where each time it accepts as
 an argument a pixel of the image with Gaussian noise and gradually through the weights w calculates and returns the corresponding pixel that will have the denoised image. 
The "patch (int i, int j, ...)" function returns the square neighborhood centered on the "pixel (i, j)" of the image.

# Parallel implementation with cuda
In the parallel implementation the complexity from N^4 drops to N^2 through the functions __global__ void CudaPatch() and __global__ void nonLocalMeans(). These functions replace the for loops in the main where the patching and denoising were done.

# Experimental results
![εικόνα](https://user-images.githubusercontent.com/77286926/137591967-294bc8ef-46f7-4148-99c1-a48aebe8c962.png)
