
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>


#define B (128*128)
#define THREADS_PER_BLOCK 128
int N = 128,
    M = 128;
int patch_size = 7;


struct timespec start, finish, t0, t1;

__device__ void patch(double *Image, int i, int j,double *patch_array, int patch_size, int N, int M);
void G(double *G_array,int patch_size);
double* addNoise(double* Image, int imSize);
__global__ void nonLocalMeans(double *newImage,double *Image, int N, int M,double *patches,int patch_size, double *G_array);
__device__ double denoise(double *Image, double *patches, int patch_size, int N, int M, int id,double *G_array);
__global__ void CudaPatch(int patch_size,double *Image, int N, int M, double *patches);


double* readFile(char* filename, int N, int M){

    double* A = (double *)malloc(N *M  * sizeof(double));

    FILE *f = fopen(filename, "r");

    for(int i = 0; i < N * M; i++) fscanf(f, "%lf %*c", &A[i]);

    fclose(f);
    return A;
}

void writeFile(double *A, char* filename, int N, int M){

    FILE *f = fopen(filename, "w");

    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < M - 1; j++)
        {
            fprintf(f, "%lf,", A[i*N + j]);
        }
        fprintf(f, "%lf", A[i*N + M-1]);
        fprintf(f, "\n");
    }

    fclose(f);

}


int main(){

    


    double *Image =readFile("cat.txt", N, M);

    


    printf("adding noise...\n");
    Image = addNoise( Image, N*N);   //image with noise

    writeFile(Image, "noise.txt" , N, M);
    printf("noise added\n");
    
    
    int squers =  patch_size*patch_size;
    

    clock_gettime(CLOCK_REALTIME, &start);

    double *newImage = (double *)malloc(N*M * sizeof(double));
    double *G_array = (double *)malloc(patch_size*patch_size * sizeof(double));
    G(G_array, patch_size);
    


    double *d_Image, *d_newImage, *d_patches, *d_G_array;
    cudaMalloc((void**)&d_Image, N*M * sizeof(double));
    cudaMalloc((void**)&d_newImage, N*M * sizeof(double));
    cudaMalloc((void**)&d_patches, N*M * squers * sizeof(double));
    cudaMalloc((void**)&d_G_array, squers * sizeof(double));

    cudaMemcpy(d_Image,  Image,  N*M * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_G_array, G_array, squers * sizeof(double), cudaMemcpyHostToDevice);
    
    CudaPatch<<<B/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(patch_size, d_Image, N, M, d_patches);
    nonLocalMeans<<<B/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(d_newImage, d_Image, N, M, d_patches, patch_size, d_G_array);
    cudaDeviceSynchronize();

    clock_gettime(CLOCK_REALTIME, &finish);


    cudaMemcpy(newImage, d_newImage, N*M * sizeof(double), cudaMemcpyDeviceToHost);
    writeFile(newImage, "denoise.txt" , N, M);
    double duration = ((finish.tv_sec - start.tv_sec) * 1000000 + (finish.tv_nsec - start.tv_nsec) / 1000) / 1000000.0;
    printf("Duration: %f from %d , %d \n", duration,N,patch_size);

    double *differences  = (double *)malloc(N*M*sizeof(double));
    for(int i=0; i<N*M; i++) differences [i]=fabs( -newImage[i] + Image[i]);
    writeFile(differences , "diference.txt" , N, M);
    return 0;

}
__global__ void CudaPatch(int patch_size,double *Image, int N, int M, double *patches){
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    patch(Image, blockIdx.x,threadIdx.x,(patches + patch_size*patch_size*id), patch_size,N,M);
}
__global__ void nonLocalMeans(double *newImage,double *Image, int N, int M,double *patches,int patch_size, double *G_array){

    int id = threadIdx.x + blockDim.x * blockIdx.x;
    newImage[id] = denoise(Image, patches,  patch_size,  N,  M,  id,G_array);

}

__device__ void patch(double *Image, int i, int j,double *patch_array, int patch_size, int N, int M){
    int p=0, o=0;
    for(int u=i-patch_size/2; u<=i+patch_size/2; u++){
        for(int v=j-patch_size/2; v<=j+patch_size/2; v++){
            if(u>=0 && v>=0 && u<N && v<M) patch_array[p++] = Image[u*N +v];
            else patch_array[p++] = Image[abs(v)*M +abs(u)];
        }
    }
}

__device__ double denoise(double *Image, double *patches, int patch_size, int N, int M, int id,double *G_array){
    
    int squers = patch_size*patch_size;
    int pointer = id*squers;
    double sigma = 0.05;
    double z=0;
    double patch_Distances = 0, newPixel=0;
    
    for(int u=0; u<N*M; u++){
        for(int v=0; v<squers; v++){
            //if(patches[pointer + v] == -1 || patches[u*squers +v] == -1) continue;
            patch_Distances += (patches[pointer + v] - patches[u*squers +v])*(patches[pointer + v] - patches[u*squers +v])*G_array[v];
        }
        double w = exp(-patch_Distances/pow(sigma,2));
        z += w;
        newPixel += w*Image[u];
        patch_Distances = 0;
    }

    return newPixel/z;
}

void G(double *G_array,int patch_size){

    int *array = (int *)malloc(patch_size*patch_size*2 * sizeof(int));

    double sigma = 5/3.0;
    int p=0;
    for(int i=-patch_size/2; i<=patch_size/2; i++){
        for(int j=-patch_size/2; j<=patch_size/2; j++){
            array[p++] = i;
            array[p++] = j;
        }
    }
    
    p=0;
    double z=0;
    for( ; p<2*patch_size*patch_size; ){
        z+=exp(-(pow(array[p++],2) + pow(array[p++],2))/(2*sigma*sigma));
    }

    p=0;
    for( int i=0; p<2*patch_size*patch_size; i++){
        G_array[i] = (1/(sqrt(2*M_PI)*sigma))*exp(-(pow(array[p++],2) + pow(array[p++],2))/(2*sigma*sigma));
    }
    
}

double* addNoise(double* Image, int imSize){

    double *noise = (double *)malloc(imSize * sizeof(double));
    double sigma = 2,value, effect;

    for(int i = 0; i < imSize; i++)
    {
        value    = ((double)( rand() ) / RAND_MAX*20 - 10);
        effect   = (1 / (sigma*sqrt(2*M_PI)))*exp((-value*value) / (2*sigma*sigma)) - 0.1;
        noise[i] = (0.5*effect + 1) * Image[i];
    }
    
    return noise;
}