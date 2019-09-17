/******************************************************************************
 * Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include "nv_wavenet.cuh"
#include <cuda_profiler_api.h>
#include <stdio.h>
#include <vector>
#include <unistd.h>

#define kR 64
#define kA 256

template <int R, int A>
float getSampleRate(int num_layers, int max_dilation, int batch_size, int batch_size_per_block, int num_samples, int num_samples_per_chunk) {

    // Set up initial activations
    int conditioning_size = num_samples * num_layers * batch_size * 2 * R * sizeof(float);
    float* conditioning = new float[conditioning_size];

    if (conditioning == NULL) {
        fprintf(stderr, "\nERROR: Unable to allocate conditioning vectors.  Try running with fewer timesteps (-n)\n\n");        
        assert(false);
    }


    std::vector<float> randomSelector(batch_size*num_samples);
    for (int i=0; i<batch_size*num_samples; i++) {
        randomSelector[i] = (float) rand() / RAND_MAX;
    }

    float *randomWeights = new float[A*A*2];
    for (int i=0; i<A*A*2; i++) {
        randomWeights[i] = -0.5 + static_cast <float> (rand()) / static_cast <float> (RAND_MAX); 
    }

    nvWavenetInfer<R, A> infer(num_layers, max_dilation, batch_size, num_samples);
    for (int l=0; l<num_layers; l++) {
        infer.setLayerWeights(l, randomWeights, randomWeights, randomWeights, randomWeights, randomWeights);
    }
    infer.setOutWeights(randomWeights, randomWeights);
    infer.setInputs(conditioning, &randomSelector[0]); 
    gpuErrChk(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    gpuErrChk(cudaEventCreate(&start));
    gpuErrChk(cudaEventCreate(&stop));
    gpuErrChk(cudaEventRecord(start));
    int* mcYout;
    // because the chunked version copies repeatedly, we should measure it as well.
    gpuErrChk(cudaMallocHost(&mcYout, num_samples*batch_size*sizeof(int)));
    cudaProfilerStart();
    bool success = infer.run_chunks(num_samples_per_chunk, [](int*, int, int){}, num_samples, batch_size, mcYout, batch_size_per_block);
    gpuErrChk(cudaFreeHost(mcYout));

    gpuErrChk(cudaEventRecord(stop));

    gpuErrChk(cudaEventSynchronize(stop));
    float elapsed_time_ms;
    gpuErrChk(cudaEventElapsedTime(&elapsed_time_ms, start, stop));
    gpuErrChk(cudaDeviceSynchronize());

    delete conditioning;
    delete randomWeights;
    return success ? float(num_samples) / elapsed_time_ms : 0.f;

}

int main(int argc, char* argv[]) {


    int num_layers = 8;
    int batch_size = 1;
    int batch_size_per_block = 1;
    int num_samples = 16384; 
    int max_dilation = 512;
    int num_samples_per_chunk = 2048;
    int device = 0;

    int c;
    while ((c = getopt (argc, argv, "l:b:n:c:d:t:f:")) != -1) {
        switch (c) {
            case 'l':
                num_layers = atoi(optarg);
                break;
            case 'b':
                batch_size = atoi(optarg);
                break;
            case 'n':
                num_samples = atoi(optarg);
                break;
            case 'c':
                batch_size_per_block = atoi(optarg);
                break;
            case 'd':
                max_dilation = atoi(optarg);
                break;
            case 't':
                num_samples_per_chunk = atoi(optarg);
                break;
            case 'f':
                device = atoi(optarg);
                break;
            default:
                assert(false);
        }
    }

    printf("R: %d\n", kR);
    printf("A: %d\n", kA);
    printf("num layers: %d\n", num_layers);
    printf("max dilation: %d\n", max_dilation);
    printf("batch size: %d\n", batch_size);
    printf("batch size per block: %d\n", batch_size_per_block);
    printf("num samples: %d\n", num_samples);

    srand(1);
    cudaSetDevice(device);

    float sample_rate = getSampleRate<kR,kA>(num_layers, max_dilation, batch_size, batch_size_per_block, num_samples, num_samples_per_chunk);
    printf("Sample rate: %f kHz\n", sample_rate);
}
