# non-local-means--image-denoising

The main idea of this problem is to denoise images with Gaussian noise input, using the non-local-mean algorithm. The non-local means filtering takes a mean of all pixels in the image, weighted by how similar these pixels are to the target pixel.


# Serial implementation
The denoise() function is called N * N times in main () where each time it accepts as an argument a pixel of the image with Gaussian noise and gradually through the weights w calculates and returns the corresponding pixel that will have the denoised image. 
-The "patch (..., int i, int j, ...)" function returns the square neighborhood centered on the "pixel (i, j)" of the image.

# Parallel implementation with cuda
In the parallel implementation the complexity from N^4 drops to N^2 through the functions __global__ void CudaPatch() and __global__ void nonLocalMeans(). These functions replace the for loops in the main where the patching and denoising were done.

# Experimental results
![Screenshot_24](https://user-images.githubusercontent.com/77286926/137591721-8ac02213-061c-4f63-a8e9-832bd91dd4fe.png)
![Screenshot_25](https://user-images.githubusercontent.com/77286926/137591722-d5f991a6-1639-4d0a-acd0-eceaa245f5dd.png)
![64_3](https://user-images.githubusercontent.com/77286926/137591723-61e7eb96-fce3-468e-b9b8-bc023e694c94.png)
