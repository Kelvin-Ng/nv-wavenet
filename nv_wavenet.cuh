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

#include "cuda_fp16.h"
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <vector>
#include <algorithm>

#include "matrix_math.cuh"
#include "softmax.cuh"
#include "nv_wavenet_util.cuh"

struct nv_wavenet_params {
    int num_samples;
    int num_samples_per_chunk;
    int blocks_per_sample;
    int init_sample;
    int batch_size;
    int num_layers;
    int* yInPrev;
    int* yInCur;
    float* embedPrev;
    float* embedCur;
    bool tanhEmbed;
    float* Wprev;
    float* L;
    float* Wcur;
    float* B;
    float* Wres;
    float* Bres;
    float* xt;
    float* xtmd;
    float* xtOut;
    float* a_prev;
    float* Wout;
    float* Bout;
    float* out;
    float* p;
    float* outputSelectors;
    int* yOut;
    bool dumpActivations;
    int maxDilation;

    float* h;
    volatile int*    ySample;

};

#include "nv_wavenet_persistent.cuh"

__global__ void silenceInputs(int* yInPrev, int* yInCur, int size) {
    for (int i=threadIdx.x; i<size; i += blockDim.x) {
        yInPrev[i] = 128;
        yInCur[i] = 128;
    }
}

template <int R=64, int A=256>
class nvWavenetInfer {
    protected:
        int m_numLayers;
        int m_maxBatch; 

        int* m_yOut;
        float* m_outputSelectors;

        float* m_embedPrev;
        float* m_embedCur;
        bool m_tanhEmbed;

        float* m_Wprev;
        float* m_Wcur;
        float* m_Wres;

        float* m_Bh;
        float* m_Lh;
        float* m_Bres;

        float* m_XtIn;
        float* m_hOut;
        float* m_aPrev;
        int* m_yInPrev;
        int* m_yInCur;

        float* m_XtOut;

        float* m_Wout;
        float*   m_Bout;

        float* m_out;
        float* m_p;

        // For dual-block
        float* m_h;
        int*    m_ySample;

        int m_maxDilation;

        int m_maxSamples;
        int m_num_samples_per_chunk;

        void setActivation(float* dst, float* src, size_t size) {
            gpuErrChk(cudaMemcpy(dst, src, size*sizeof(float), cudaMemcpyDefault));
        }
        void getActivation(float* dst, float* src, size_t size) {
            gpuErrChk(cudaMemcpy(dst, src, size*sizeof(float), cudaMemcpyDefault));
        }
        void setLayerWeight(int layer, float* dst, float* src, int M, int K) {
            gpuErrChk(cudaMemcpy(dst + layer*M*K, src, M*K*sizeof(float), cudaMemcpyDefault));
        }
        void setLayerBias(int layer, float* dst, float* src, int M){
            gpuErrChk(cudaMemcpy(dst + layer*M, src, M*sizeof(float), cudaMemcpyDefault));
        }

    public:
        nvWavenetInfer (int numLayers, int maxDilation, int batchSize, int numSamples, bool tanhEmbed=true) : m_numLayers(numLayers), m_maxBatch(batchSize), m_maxSamples(numSamples), m_tanhEmbed(tanhEmbed) {

            m_num_samples_per_chunk = 0;
            m_maxDilation = maxDilation;

            gpuErrChk(cudaMalloc(&m_yOut, numSamples*batchSize*sizeof(int))); // one-hot vector represented as single value indicating which value is set
            gpuErrChk(cudaMemset(m_yOut, 0, numSamples*batchSize*sizeof(int))); 
            gpuErrChk(cudaMalloc(&m_outputSelectors, numSamples*batchSize*sizeof(float))); 

            gpuErrChk(cudaMalloc(&m_embedPrev, A*R*sizeof(float)));
            gpuErrChk(cudaMalloc(&m_embedCur, A*R*sizeof(float)));

            gpuErrChk(cudaMalloc(&m_Wprev, numLayers*2*R*R*sizeof(float)));
            gpuErrChk(cudaMalloc(&m_Wcur, numLayers*2*R*R*sizeof(float)));
            gpuErrChk(cudaMalloc(&m_Bh, numLayers*2*R*sizeof(float)));
            gpuErrChk(cudaMalloc(&m_Lh, numSamples*numLayers*batchSize*2*R*sizeof(float)));
            gpuErrChk(cudaMalloc(&m_Wres, numLayers*R*R*sizeof(float)));
            gpuErrChk(cudaMalloc(&m_Bres, numLayers*R*sizeof(float)));
            gpuErrChk(cudaMalloc(&m_XtOut, numLayers*R*batchSize*sizeof(float)));

            // For now, just burn memory as though all layers had the maximum dilation value
            gpuErrChk(cudaMalloc(&m_XtIn, (m_maxDilation+1)*(numLayers+1)*R*batchSize*sizeof(float)));
            gpuErrChk(cudaMalloc(&m_hOut, numLayers*batchSize*R*sizeof(float)));
            gpuErrChk(cudaMalloc(&m_aPrev, numLayers*batchSize*2*R*sizeof(float)));
            gpuErrChk(cudaMalloc(&m_yInPrev, batchSize*sizeof(int))); // one-hot vector represented as single value indicating which value is set
            gpuErrChk(cudaMalloc(&m_yInCur, batchSize*sizeof(int))); // one-hot vector represented as single value indicating which value is set

            gpuErrChk(cudaMalloc(&m_Wout, A*A*sizeof(float)));
            gpuErrChk(cudaMalloc(&m_Bout, A*sizeof(float)));
            gpuErrChk(cudaMalloc(&m_out, A*batchSize*A/R*sizeof(float)));
            gpuErrChk(cudaMalloc(&m_p, A*batchSize*sizeof(float)));

            gpuErrChk(cudaMalloc(&m_h, numLayers*batchSize*R*sizeof(float)));
            gpuErrChk(cudaMalloc(&m_ySample, batchSize*sizeof(int)));
        }
        ~nvWavenetInfer() {
            gpuErrChk(cudaFree(m_yOut));
            gpuErrChk(cudaFree(m_outputSelectors));
            gpuErrChk(cudaFree(m_embedPrev));
            gpuErrChk(cudaFree(m_embedCur));
            gpuErrChk(cudaFree(m_Wprev));
            gpuErrChk(cudaFree(m_Wcur));
            gpuErrChk(cudaFree(m_Bh));
            gpuErrChk(cudaFree(m_Lh));
            gpuErrChk(cudaFree(m_Wres));
            gpuErrChk(cudaFree(m_Bres));
            gpuErrChk(cudaFree(m_XtOut));
            gpuErrChk(cudaFree(m_XtIn));
            gpuErrChk(cudaFree(m_hOut));
            gpuErrChk(cudaFree(m_aPrev));
            gpuErrChk(cudaFree(m_yInPrev));
            gpuErrChk(cudaFree(m_yInCur));
            gpuErrChk(cudaFree(m_Wout));
            gpuErrChk(cudaFree(m_Bout));
            gpuErrChk(cudaFree(m_out));
            gpuErrChk(cudaFree(m_p));
        }
        virtual void setEmbeddings (float* embedPrev, float* embedCur) {
            setActivation(m_embedPrev, embedPrev, A*R);
            setActivation(m_embedCur, embedCur, A*R);
        }
        virtual void setLayerWeights (int layer, float* Wprev, float* Wcur, float* Bh, float* Wres, float* Bres) {
            setLayerWeight(layer, m_Wprev, Wprev, 2*R, R);
            setLayerWeight(layer, m_Wcur, Wcur, 2*R, R);
            setLayerWeight(layer, m_Wres, Wres, R, R);

            setLayerBias(layer, m_Bh, Bh, 2*R);
            setLayerBias(layer, m_Bres, Bres, R);
        }
        virtual void setOutWeights (float* Wza, float* Bza) {
            setLayerWeight(0, m_Wout, Wza, A, A);
            setLayerBias(0, m_Bout, Bza, A);
        }

        void setInputs (float* Lh, float* outputSelectors) {
            silenceInputs<<<1,256>>>(m_yInPrev, m_yInCur, m_maxBatch);
            setActivation(m_Lh, Lh, m_maxSamples*m_numLayers*m_maxBatch*2*R);
            gpuErrChk(cudaMemcpy(m_outputSelectors, outputSelectors, m_maxSamples*m_maxBatch*sizeof(float), cudaMemcpyHostToDevice));

        }

        void getXtOut(int layer, float* hXt) { getActivation(hXt, m_XtOut + layer*m_maxBatch*R, m_maxBatch*R); }
        void getP(float* hP) { getActivation(hP, m_p, m_maxBatch*A); }
        void getYOut(int* yOut, int offset, int size, cudaStream_t stream = 0) {
            size_t cpy_pitch = m_maxSamples * sizeof(int); // spacing between chunk first elements
            size_t cpy_width = size * sizeof(int); // size of individual chunk
            size_t cpy_height = m_maxBatch;
            gpuErrChk(cudaMemcpy2DAsync(yOut + offset, cpy_pitch, m_yOut + offset, cpy_pitch, cpy_width, cpy_height, cudaMemcpyDeviceToHost, stream));
        }
        template<class Callback>
        bool run_chunks(int num_samples_per_chunk, Callback consume, int num_samples, int batch_size, int* yOut=NULL, int batch_size_per_block=1, bool dumpActivations=false, cudaStream_t stream = 0) {
            bool result = true;
            cudaStream_t stream_compute, stream_copy;
            if(!stream)
              cudaStreamCreate(&stream_compute);
            else
              stream_compute = stream;
            cudaStreamCreate(&stream_copy);
            m_num_samples_per_chunk = num_samples_per_chunk;
            int num_chunks = (num_samples + m_num_samples_per_chunk - 1) / m_num_samples_per_chunk;

            std::vector<cudaEvent_t> event_compute(num_chunks);
            std::vector<cudaEvent_t> event_copy(num_chunks);
            for (int j = 0; j < num_chunks; j++) {
              cudaEventCreateWithFlags(&(event_compute[j]), cudaEventDisableTiming);
              cudaEventCreateWithFlags(&(event_copy[j]), cudaEventDisableTiming);
            }

            for (int j = 0; j < num_chunks; j++) {

              int initSample = j*m_num_samples_per_chunk;
              if (j == num_chunks - 1) {
                m_num_samples_per_chunk = num_samples - initSample;
              }

              result = result && run_partial(initSample, num_samples, batch_size, NULL, batch_size_per_block, true, stream_compute);
              cudaEventRecord(event_compute[j], stream_compute);
              cudaStreamWaitEvent(stream_copy, event_compute[j], 0);
              if(yOut != NULL)
                getYOut(yOut, initSample, m_num_samples_per_chunk, stream_copy);
              cudaEventRecord(event_copy[j], stream_copy);
            }
            m_num_samples_per_chunk = num_samples_per_chunk;
            for (int j = 0; j < num_chunks; j++) {

              int initSample = j*m_num_samples_per_chunk;
              if (j == num_chunks - 1) {
                m_num_samples_per_chunk = num_samples - initSample;
              }
              cudaEventSynchronize(event_copy[j]);
              consume(yOut, initSample, m_num_samples_per_chunk);
            }
            m_num_samples_per_chunk = 0;
            for (int j = 0; j < num_chunks; j++) {
              cudaEventDestroy(event_compute[j]);
              cudaEventDestroy(event_copy[j]);
            }
            if(stream != stream_compute)
              cudaStreamDestroy(stream_compute);
            cudaStreamDestroy(stream_copy);
            return result;
        }

        bool run_partial(int init_sample, int num_samples, int batch_size, int* yOut=NULL, int batch_size_per_block=1, bool dumpActivations=false, cudaStream_t stream = 0) {
            nv_wavenet_params params;
            params.num_samples = num_samples;
            params.init_sample = init_sample;
            params.num_samples_per_chunk = m_num_samples_per_chunk ? m_num_samples_per_chunk : num_samples;
            params.batch_size = batch_size;
            params.num_layers = m_numLayers;
            params.yInPrev = m_yInPrev;
            params.yInCur = m_yInCur;
            params.embedPrev = m_embedPrev;
            params.embedCur = m_embedCur;
            params.tanhEmbed = m_tanhEmbed;
            params.Wprev = m_Wprev;
            params.L = m_Lh;
            params.Wcur = m_Wcur;
            params.B = m_Bh;
            params.Wres = m_Wres;
            params.Bres = m_Bres;
            params.xt = m_XtIn;
            params.xtOut = m_XtOut;
            params.a_prev = m_aPrev;
            params.Wout = m_Wout;
            params.Bout = m_Bout;
            params.out = m_out;
            params.p = m_p;
            params.outputSelectors = m_outputSelectors;
            params.yOut = m_yOut;
            params.dumpActivations = dumpActivations;
            params.maxDilation = m_maxDilation;

            params.h = m_h;
            params.ySample = m_ySample;

            bool result = false;

            assert(batch_size_per_block < 5);
            if (batch_size_per_block == 4) {
                assert(batch_size%4==0);
                result = launch_persistent<R, 4>()(params, stream);
            }
            else if (batch_size_per_block == 3) {
                assert(batch_size%3==0);
                result =  launch_persistent<R, 3>()(params, stream);
            }
            else if (batch_size_per_block == 2) {
                assert(batch_size%2==0);
                result =  launch_persistent<R, 2>()(params, stream);
            }
            else {
                result =  launch_persistent<R, 1>()(params, stream);
            }
            if (yOut != NULL) {
                gpuErrChk(cudaMemcpyAsync(yOut, m_yOut, m_maxSamples*m_maxBatch*sizeof(int), cudaMemcpyDeviceToHost, stream));
            }
            return result;
        }
        bool run(int num_samples, int batch_size, int* yOut=NULL, int batch_size_per_block=1, bool dumpActivations=false, cudaStream_t stream = 0) {
            m_num_samples_per_chunk = 0;
            return run_partial(0, num_samples, batch_size, yOut, batch_size_per_block, dumpActivations, stream);
        }
};

