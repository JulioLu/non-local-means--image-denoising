
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>

struct timespec start, finish, t0, t1;

void patch(float *Image, int i, int j,float *patch_array, int patch_size, int N, int M);
void G(float *G_array,int patch_size);
float* addNoise(float* Image, int imSize);
float denoise(float *Image, float *patches, int patch_size, int N, int M, int id,float *G_array);


float* readFile(char* filename, int N, int M){

    float* A = (float *)malloc(N *M  * sizeof(float));

    FILE *f = fopen(filename, "r");

    for(int i = 0; i < N * M; i++) fscanf(f, "%f %*c", &A[i]);

    fclose(f);
    return A;
}

void writeFile(float *A, char* filename, int N, int M){

    FILE *f = fopen(filename, "w");

    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < M - 1; j++)
        {
            fprintf(f, "%f,", A[i*N + j]);
        }
        fprintf(f, "%f", A[i*N + M-1]);
        fprintf(f, "\n");
    }

    fclose(f);

}


int main(){

    
    int N = 64,
        M = 64;


    float *Image =readFile("house.txt", N, M);

    


    printf("adding noise...\n");
    Image = addNoise( Image, N*N);   //image with noise

    writeFile(Image, "noise.txt" , N, M);
    printf("noise added\n");
    clock_gettime(CLOCK_REALTIME, &start);
    int patch_size = 5;

    int squers = patch_size*patch_size;
    float *patches = (float *)malloc(N*M*patch_size*patch_size * sizeof(float*));

    int l=0;
    for(int i=0; i<N; i++)
        for(int j=0; j<M; j++)
            patch(Image, i,j,(patches + squers*l++), patch_size,N,M);

    
    float *newImage = (float *)malloc(N*M * sizeof(float));
    float *G_array = (float *)malloc(patch_size*patch_size * sizeof(float));
    G(G_array, patch_size);


    for(int i=0; i<N*M; i++) newImage[i] = denoise(Image, patches, patch_size, N, M, i, G_array);

    clock_gettime(CLOCK_REALTIME, &finish);


    writeFile(newImage, "denoiseImage.txt" , N, M);
    float duration = ((finish.tv_sec - start.tv_sec) * 1000000 + (finish.tv_nsec - start.tv_nsec) / 1000) / 1000000.0;
    printf("Duration: %f\n", duration);

    float *differences  = (float *)malloc(N*M*sizeof(float));
    for(int i=0; i<N*M; i++) differences [i]=fabs( -newImage[i] + Image[i]);
    writeFile(differences , "diference.txt" , N, M);
    return 0;

}


void patch(float *Image, int i, int j,float *patch_array, int patch_size, int N, int M){
    int p=0, o=0;
    for(int u=i-patch_size/2; u<=i+patch_size/2; u++){
        for(int v=j-patch_size/2; v<=j+patch_size/2; v++){
            if(u>=0 && v>=0 && u<N && v<M) patch_array[p++] = Image[u*N +v];
            else patch_array[p++] = Image[abs(v)*M +abs(u)];
        }
    }
}

float denoise(float *Image, float *patches, int patch_size, int N, int M, int id,float *G_array){
    
    int squers = patch_size*patch_size;
    int pointer = id*squers;
    float sigma = 0.04;
    float z=0;
    float patch_Distances = 0, newPixel=0;
    
    for(int u=0; u<N*M; u++){
        for(int v=0; v<squers; v++){
            patch_Distances += (patches[pointer + v] - patches[u*squers +v])*(patches[pointer + v] - patches[u*squers +v])*G_array[v];
        }
        float w = exp(-patch_Distances/pow(sigma,2));
        z += w;
        newPixel += w*Image[u];
        patch_Distances = 0;
    }

    return newPixel/z;
}

void G(float *G_array,int patch_size){

    int *array = (int *)malloc(patch_size*patch_size*2 * sizeof(int));

    float sigma = 5/3.0;
    int p=0;
    for(int i=-patch_size/2; i<=patch_size/2; i++){
        for(int j=-patch_size/2; j<=patch_size/2; j++){
            array[p++] = i;
            array[p++] = j;
        }
    }
    
    p=0;
    float z=0;
    for( ; p<2*patch_size*patch_size; ){
        z+=exp(-(pow(array[p++],2) + pow(array[p++],2))/(2*sigma*sigma));
    }

    p=0;
    for( int i=0; p<2*patch_size*patch_size; i++){
        G_array[i] = (1/(sqrt(2*M_PI)*sigma))*exp(-(pow(array[p++],2) + pow(array[p++],2))/(2*sigma*sigma));
    }
    //float max = G_array[patch_size*patch_size/2];
    //for(int i=0; i<patch_size*patch_size; i++) G_array[i] = G_array[i]/max;
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