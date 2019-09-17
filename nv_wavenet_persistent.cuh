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

#define PERS_SOFTMAX_BLOCKS 4

__device__ __forceinline__ bool isNegativeZero(float a) {
    int ret;
    asm volatile("{  set.eq.s32.b32 %0, %1, %2;}\n" : "=r"(ret) : "f"(a), "r"(0x80000000));
    return ret;
}

__device__ __forceinline__ float validate(float a) {
    return isNegativeZero(a) ? 0.f : a;
}

__device__ __forceinline__ void storeValidate(volatile float* y, int index, float val) {
    y[index] = validate(val);
}

template <int R>
__global__ void initializeActivations(float* xt, float* h_out, float* a_prev, int num_layers, int batch_size) {
    assert(blockDim.x == R);

    int offset = blockIdx.x*blockDim.x + threadIdx.x;

    xt[offset] = -0.f;
    h_out[offset] = -0.f;

    a_prev[offset*2] = -0.f;
    a_prev[offset*2 + 1] = -0.f;
}

__global__ void initializeActivationsGeneric(float* skipIn) {
    int offset = blockIdx.x*blockDim.x + threadIdx.x;
    skipIn[offset] = -0.f;
}

// Make sure all necessary clears are completed before processing a new sample.  Lock is per batch index.
template <int BATCH_UNROLL> 
__device__ __inline__ void sampleLockAcquire(int batch_offset, int sample, volatile int* sampleLock){
    if (threadIdx.x == 0) {
        bool valid = false;
        while (!valid) {
            valid = true;
#pragma unroll
            for (int u=0; u<BATCH_UNROLL; u++) {  
                valid &= (sampleLock[batch_offset+u]>=sample);
            }
        }
    }
    __syncthreads();
}

template <int R, int BATCH_UNROLL>
__device__ void nv_wavenet_persistent_prev(int row, int num_samples, int init_sample, int num_samples_per_chunk, volatile int* ySample, int layer, int num_layers, int batch_size, int maxDilation, float* Wprev, float* a_prev, volatile float* xt) {
    float weights[R];
    loadWeights<2*R,R>(weights,Wprev,layer,row);
    float accum[BATCH_UNROLL];
    __shared__ float xtmd_sh[BATCH_UNROLL][R];

    int dilation = 1;
    for (int l=1; l<=layer; l++) {
        dilation = dilation << 1;
        if (dilation > maxDilation) dilation = 1;
    }

    if (row < 2*R) {
        for (int sample=init_sample; sample<init_sample+num_samples_per_chunk; sample++) {
            int sample_offset = (sample - dilation) % (maxDilation+1);
            volatile float* xtmd = xt + sample_offset*(num_layers+1)*R*batch_size;
            for (int batch_offset = 0; batch_offset < batch_size; batch_offset += BATCH_UNROLL) {
                sampleLockAcquire<BATCH_UNROLL>(batch_offset,sample,ySample);
                if (row < R) {
#pragma unroll
                    for (int b=0; b<BATCH_UNROLL; b++) {
                        xtmd_sh[b][row] = (dilation <= sample) ? loadVolatile(xtmd,layer*batch_size*R + (batch_offset+b)*R + row) : (float)0.f;
                    }
                }
                __syncthreads();
                GEMM<R,2,BATCH_UNROLL>(weights, xtmd_sh, accum);
#pragma unroll
                for (int b=0; b<BATCH_UNROLL; b++) {
                    a_prev[layer*batch_size*2*R + (batch_offset+b)*2*R + threadIdx.x] = accum[b]; 
                }
            }
        }
    }
}

template <int R, int BATCH_UNROLL>
__device__ void nv_wavenet_persistent_cur(int row, int num_samples, int init_sample, int num_samples_per_chunk, volatile int* ySample, int layer, int num_layers, int batch_size, int maxDilation, float* Wcur, float* B, float* L, float a_cur_sh[BATCH_UNROLL][2*R], volatile float* a_prev, volatile float* xt, int* yInPrev, int* yInCur, float* embedPrev, float* embedCur, bool tanhEmbed) {
    float weights[R];
    loadWeights<2*R,R>(weights,Wcur,layer,row);
    float accum[BATCH_UNROLL];
    float bias = B[layer*2*R+row];
    float a_prev_reg[BATCH_UNROLL];
    float xt_in[BATCH_UNROLL];

    for (int sample=init_sample; sample<init_sample+num_samples_per_chunk; sample++) {
        __syncthreads(); // Wait for initial sample lock
        volatile float* Xt = xt + (sample%(maxDilation+1))*(num_layers+1)*R*batch_size;
        for (int batch_offset = 0; batch_offset < batch_size; batch_offset += BATCH_UNROLL) {
            float conditioning[BATCH_UNROLL];
#pragma unroll
            for (int b=0; b<BATCH_UNROLL; b++) {
                conditioning[b] = L[sample*num_layers*batch_size*2*R + layer*batch_size*2*R + (batch_offset+b)*2*R + row];
            }
            __shared__ float xt_sh[BATCH_UNROLL][R];
            if (row < R) {
                if (layer == 0) {
                    // Embedding
                    int yPrev[BATCH_UNROLL];
                    int yCur[BATCH_UNROLL];
#pragma unroll
                    for (int b=0; b<BATCH_UNROLL; b++) {
                        yPrev[b] = yInPrev[batch_offset+b];
                        yCur[b] = yInCur[batch_offset+b];
                        float embedded = embedPrev[yPrev[b]*R + row] + embedCur[yCur[b]*R + row];
                        if (tanhEmbed) embedded = _tanh(embedded);
                        xt_sh[b][row] = embedded;
                        storeValidate(Xt, layer*batch_size*R + (batch_offset+b)*R + row, embedded);
                    }
                    // Make Xt visible before we write h, so that clears don't race ahead
                    // This is only needed for the embedding write, since it's read by the same block -- 
                    //  all other Xt writes get read by different blocks before they write h.  Since
                    //  the clears depend on h, then we know that the Xt writes are globally visible.
                    __threadfence();
                }
            }
            bool valid = false;
            int a_prev_offset = layer*batch_size*2*R + batch_offset*2*R + row;
            int xt_offset = layer*batch_size*R + batch_offset*R + row;
            // Do redundant loads in upper half to avoid branch in polling loop.
            if (row >= R) xt_offset -= R;
            while (!valid) {
                valid = true;
#pragma unroll
                for (int b=0; b<BATCH_UNROLL; b++) {
                    a_prev_reg[b] = loadVolatile(a_prev,a_prev_offset+b*2*R);
                    xt_in[b] = loadVolatile(Xt,xt_offset+b*R);
                }
#pragma unroll
                for (int b=0; b<BATCH_UNROLL; b++) {
                    valid &= !isNegativeZero(a_prev_reg[b]);
                    valid &= !isNegativeZero(xt_in[b]);
                }
            }
            if (row < R) {
#pragma unroll
                for (int b=0; b<BATCH_UNROLL; b++) {
                    xt_sh[b][row] = xt_in[b];
                }
            }
            namedBarrierSync(1,2*R);
            GEMM<R,2,BATCH_UNROLL>(weights,xt_sh,accum);
#pragma unroll
            for (int b=0; b<BATCH_UNROLL; b++) { 
                accum[b] += a_prev_reg[b];
                accum[b] += bias; 
                accum[b] += conditioning[b];
                float val = (row < R) ? _tanh(accum[b]) : sigmoid(accum[b]);
                a_cur_sh[b][row] = val;
            }
            namedBarrierSync(3,3*R); // a_cur_sh produced
            __syncthreads(); // a_cur_sh consumed
        }
    }
}

template <int R, int BATCH_UNROLL>
__device__ void nv_wavenet_persistent_res(int row, int num_samples, int init_sample, int num_samples_per_chunk, volatile int* ySample, int layer, int num_layers, int batch_size, int maxDilation, float* Wres, float* Bres, float a_cur_sh[BATCH_UNROLL][2*R], float* xt, float* h, float* xtOut, bool dumpActivations) {
    float weights[R];
    float bias = Bres[layer*R+row];
    float accum[BATCH_UNROLL];
    __shared__ float h_sh[BATCH_UNROLL][R];
    loadWeights<R,R>(weights,Wres,layer,row);
    for (int sample=init_sample; sample<init_sample+num_samples_per_chunk; sample++) {
        __syncthreads(); // Wait for initial sample lock
        for (int batch_offset = 0; batch_offset < batch_size; batch_offset += BATCH_UNROLL) {
            namedBarrierSync(3,3*R); // a_cur_sh produced, h_sh consumed
#pragma unroll
            for (int b=0; b<BATCH_UNROLL; b++) {
                float val = a_cur_sh[b][row] * a_cur_sh[b][row + R];
                h_sh[b][row] = val;
                h[layer*batch_size*R + (batch_offset+b)*R + row] = validate(val);
            }
            __syncthreads(); // a_cur_sh consumed, h_sh produced
            GEMM<R,2,BATCH_UNROLL>(weights,h_sh,accum);
            float* Xt = xt + (sample%(maxDilation+1))*(num_layers+1)*R*batch_size;
#pragma unroll
            for (int b=0; b<BATCH_UNROLL; b++) { 
                accum[b] += bias; 
                accum[b] += Xt[layer*batch_size*R + (batch_offset+b)*R + row];
                Xt[(layer+1)*batch_size*R + (batch_offset+b)*R + row] = accum[b];
                if (dumpActivations) xtOut[layer*batch_size*R + (batch_offset+b)*R + row] = accum[b];
            }
        }
    }
}

template <int R, int BATCH_UNROLL>
__device__ void nv_wavenet_persistent_cur_res(int thread_id, int num_samples, int init_sample, int num_samples_per_chunk, volatile int* ySample, int layer, int num_layers, int batch_size, int maxDilation, float* Wcur, float* B, float* L, float* Wres, float* Bres, float* a_prev, float* xt, float* h, float* xtOut, bool dumpActivations, int* yInPrev, int* yInCur, float* embedPrev, float* embedCur, bool tanhEmbed) {
    __shared__ float a_cur_sh[BATCH_UNROLL][2*R];
    if (thread_id < R) {
        for (int sample=init_sample; sample<init_sample+num_samples_per_chunk; sample++) {
            sampleLockAcquire<BATCH_UNROLL>(0,sample,ySample);
            for (int batch_offset = 0; batch_offset < batch_size; batch_offset += BATCH_UNROLL) {
                if (batch_offset+BATCH_UNROLL<batch_size){
                    sampleLockAcquire<BATCH_UNROLL>(batch_offset+BATCH_UNROLL,sample,ySample);
                } else {
                    __syncthreads();
                }
            }
        }
    }
    else if (thread_id < 3*R) {
        int row = thread_id - R;
        nv_wavenet_persistent_cur<R, BATCH_UNROLL>(row, num_samples, init_sample, num_samples_per_chunk, ySample, layer, num_layers, batch_size, maxDilation, Wcur, B, L, a_cur_sh, a_prev, xt, yInPrev, yInCur, embedPrev, embedCur, tanhEmbed); 
    }
    else if (thread_id < 4*R) {
        int row = thread_id - 3*R;
        nv_wavenet_persistent_res<R, BATCH_UNROLL>(row, num_samples, init_sample, num_samples_per_chunk, ySample, layer, num_layers, batch_size, maxDilation, Wres, Bres, a_cur_sh, xt, h, xtOut, dumpActivations);
    }
}

template <int R, int BATCH_UNROLL>
__global__ void nv_wavenet_persistent(nv_wavenet_params params) {
    int prev_blocks = params.num_layers;
    int cur_blocks = params.num_layers;
    int thread_id = threadIdx.x;

    int init_sample = params.init_sample;
    int block_idx = blockIdx.x;
    int num_samples_per_block = params.num_samples_per_chunk;

    if (block_idx < prev_blocks) {
        // Prev
        int layer = block_idx;
        nv_wavenet_persistent_prev<R, BATCH_UNROLL>(thread_id, params.num_samples, init_sample, num_samples_per_block, params.ySample, layer, params.num_layers, params.batch_size, params.maxDilation, params.Wprev, params.a_prev, params.xt);
    } else if (block_idx < prev_blocks + cur_blocks) {
        // Cur
        int layer = block_idx - prev_blocks;
        nv_wavenet_persistent_cur_res<R, BATCH_UNROLL>(thread_id, params.num_samples, init_sample, num_samples_per_block, params.ySample, layer, params.num_layers, params.batch_size, params.maxDilation, params.Wcur, params.B, params.L, params.Wres, params.Bres, params.a_prev, params.xt, params.h, params.xtOut, params.dumpActivations, params.yInPrev, params.yInCur, params.embedPrev, params.embedCur, params.tanhEmbed);
    } else {
        // VIN: only used for advancing locks. Output should be pulled here.
        int block_id = block_idx - prev_blocks - cur_blocks;
        for (int sample = init_sample; sample < init_sample+params.num_samples_per_chunk; sample++) {
            // Make sure all the clears are visible before we advance the sample lock
            __threadfence();
            __syncthreads();
            for (int col = block_id*BATCH_UNROLL; col < params.batch_size; col += BATCH_UNROLL) {
                if (threadIdx.x == 0) {
#pragma unroll
                    for (int u=0; u<BATCH_UNROLL; u++) {
                        params.ySample[col+u] = sample+1;
                    }
                }
            }
        }
    }
}

template <int R, int BATCH_UNROLL>
struct launch_manyblock {
    bool operator() (nv_wavenet_params params, cudaStream_t stream) {
        int prev_blocks = params.num_layers;
        int cur_blocks = params.num_layers;
        int softmax_blocks = min(PERS_SOFTMAX_BLOCKS, params.batch_size);
        dim3 grid(prev_blocks + cur_blocks + softmax_blocks);
        params.blocks_per_sample = grid.x;
        dim3 block(4*R);
        int occ = getOccupancy(0, block.x*block.y*block.z,(void*)nv_wavenet_persistent<R, BATCH_UNROLL>);
        printf("%d blocks, %d blocks per SM\n", grid.x, occ);
        assert(occ>0);

        if(!params.init_sample) {
            gpuErrChk(cudaMemset((void*)params.ySample,0,params.batch_size*sizeof(int)));
            initializeActivations<R><<<params.num_layers*params.batch_size,R,0,stream>>>(params.xt, params.h, params.a_prev, params.num_layers, params.batch_size);
            initializeActivationsGeneric<<<(params.maxDilation+1)*(params.num_layers+1)*params.batch_size,R,0,stream>>>(params.xt);
        }
        void* p_params = {&params};
        cudaError_t code;
        code = cudaLaunchCooperativeKernel((void*)nv_wavenet_persistent<R,BATCH_UNROLL>, grid, block, &p_params, 0, stream);
        gpuAssert(code, __FILE__, __LINE__, false);
        return code == cudaSuccess;
    }
};

template <int R, int BATCH_UNROLL>
struct launch_persistent {
    bool operator() (nv_wavenet_params params, cudaStream_t stream) {
        return launch_manyblock<R, BATCH_UNROLL>()(params, stream);
    }
};
