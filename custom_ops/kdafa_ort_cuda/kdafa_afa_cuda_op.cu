#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT

#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>


#define CUDA_CHECK(expr)                                                     \
    do {                                                                     \
        cudaError_t err = (expr);                                            \
        if (err != cudaSuccess) {                                            \
            throw std::runtime_error(std::string("CUDA error: ") +           \
                                     cudaGetErrorString(err));               \
        }                                                                    \
    } while (0)

#define CUFFT_CHECK(expr)                                                    \
    do {                                                                     \
        cufftResult err = (expr);                                            \
        if (err != CUFFT_SUCCESS) {                                          \
            throw std::runtime_error(std::string("cuFFT error code: ") +     \
                                     std::to_string(static_cast<int>(err)));  \
        }                                                                    \
    } while (0)


static inline size_t numel4(int64_t N, int64_t C, int64_t H, int64_t W) {
    return static_cast<size_t>(N * C * H * W);
}


__device__ __forceinline__ size_t offset4_dev(
    int64_t n, int64_t c, int64_t h, int64_t w,
    int64_t C, int64_t H, int64_t W
) {
    return static_cast<size_t>(((n * C + c) * H + h) * W + w);
}


__device__ __forceinline__ float clamp_dev(float v, float lo, float hi) {
    return fminf(hi, fmaxf(lo, v));
}


__global__ void real_to_complex_kernel(
    const float* x,
    cufftComplex* z,
    size_t total
) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx >= total) {
        return;
    }

    z[idx].x = x[idx];
    z[idx].y = 0.0f;
}


__global__ void make_low_high_phase_shift_kernel(
    const cufftComplex* fft_unshifted,
    float* low_pass,
    float* high_pass,
    float* phase,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    float cuton,
    float forward_scale
) {
    size_t total = static_cast<size_t>(N * C * H * W);
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx >= total) {
        return;
    }

    int64_t tmp = static_cast<int64_t>(idx);

    int64_t x = tmp % W;
    tmp /= W;

    int64_t y = tmp % H;
    tmp /= H;

    int64_t c = tmp % C;
    tmp /= C;

    int64_t n = tmp;

    // torch.fft.fftshift와 동일한 방향
    int64_t src_y = (y + H / 2) % H;
    int64_t src_x = (x + W / 2) % W;

    size_t src_idx = offset4_dev(n, c, src_y, src_x, C, H, W);

    float real = fft_unshifted[src_idx].x * forward_scale;
    float imag = fft_unshifted[src_idx].y * forward_scale;

    float mag = sqrtf(real * real + imag * imag);
    mag = clamp_dev(mag, 1e-6f, 1e6f);

    float ph = atan2f(imag, real);

    int64_t cy = H / 2;
    int64_t cx = W / 2;
    int64_t rh = static_cast<int64_t>(cuton * static_cast<float>(cy));
    int64_t rw = static_cast<int64_t>(cuton * static_cast<float>(cx));

    bool is_low =
        (y >= cy - rh) &&
        (y <  cy + rh) &&
        (x >= cx - rw) &&
        (x <  cx + rw);

    low_pass[idx] = is_low ? mag : 0.0f;
    high_pass[idx] = is_low ? 0.0f : mag;
    phase[idx] = ph;
}


__global__ void attention_1x1_kernel(
    const float* feature,
    const float* weight,
    float* attn,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W
) {
    size_t total = static_cast<size_t>(N * C * H * W);
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx >= total) {
        return;
    }

    int64_t tmp = static_cast<int64_t>(idx);

    int64_t x = tmp % W;
    tmp /= W;

    int64_t y = tmp % H;
    tmp /= H;

    int64_t co = tmp % C;
    tmp /= C;

    int64_t n = tmp;

    float sum = 0.0f;

    for (int64_t ci = 0; ci < C; ++ci) {
        size_t in_idx = offset4_dev(n, ci, y, x, C, H, W);
        size_t w_idx = static_cast<size_t>(co * C + ci);

        sum += weight[w_idx] * feature[in_idx];
    }

    attn[idx] = sum;
}


__global__ void reconstruct_ifft_input_kernel(
    const float* low_pass,
    const float* high_pass,
    const float* phase,
    const float* low_attn,
    const float* high_attn,
    cufftComplex* ifft_input_unshifted,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W
) {
    size_t pixel_total = static_cast<size_t>(N * H * W);
    size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (p >= pixel_total) {
        return;
    }

    int64_t tmp = static_cast<int64_t>(p);

    int64_t x = tmp % W;
    tmp /= W;

    int64_t y = tmp % H;
    tmp /= H;

    int64_t n = tmp;

    // ifftshift:
    // even size에서는 H/2, W/2와 동일
    int64_t sy = (y + (H + 1) / 2) % H;
    int64_t sx = (x + (W + 1) / 2) % W;

    float low_max = -3.402823466e+38f;
    float high_max = -3.402823466e+38f;

    for (int64_t c = 0; c < C; ++c) {
        size_t sidx = offset4_dev(n, c, sy, sx, C, H, W);

        low_max = fmaxf(low_max, low_attn[sidx]);
        high_max = fmaxf(high_max, high_attn[sidx]);
    }

    float low_sum = 0.0f;
    float high_sum = 0.0f;

    for (int64_t c = 0; c < C; ++c) {
        size_t sidx = offset4_dev(n, c, sy, sx, C, H, W);

        low_sum += expf(low_attn[sidx] - low_max);
        high_sum += expf(high_attn[sidx] - high_max);
    }

    for (int64_t c = 0; c < C; ++c) {
        size_t sidx = offset4_dev(n, c, sy, sx, C, H, W);
        size_t didx = offset4_dev(n, c, y, x, C, H, W);

        float low_sm = expf(low_attn[sidx] - low_max) / low_sum;
        float high_sm = expf(high_attn[sidx] - high_max) / high_sum;

        float mag =
            low_sm * low_pass[sidx] +
            high_sm * high_pass[sidx];

        mag = clamp_dev(mag, 1e-6f, 1e6f);

        float ph = phase[sidx];

        ifft_input_unshifted[didx].x = mag * cosf(ph);
        ifft_input_unshifted[didx].y = mag * sinf(ph);
    }
}


__global__ void complex_to_real_scale_kernel(
    const cufftComplex* z,
    float* y,
    size_t total,
    float inverse_scale
) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx >= total) {
        return;
    }

    y[idx] = z[idx].x * inverse_scale;
}


static void compute_afa_cuda(
    const float* x,
    const float* low_w,
    const float* high_w,
    float* y,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    float cuton,
    cufftHandle plan
) {
    const size_t total = numel4(N, C, H, W);
    const size_t complex_bytes = total * sizeof(cufftComplex);
    const size_t float_bytes = total * sizeof(float);

    const int threads = 256;
    const int blocks_total = static_cast<int>((total + threads - 1) / threads);
    const size_t pixel_total = static_cast<size_t>(N * H * W);
    const int blocks_pixel = static_cast<int>((pixel_total + threads - 1) / threads);

    const float ortho_scale = 1.0f / sqrtf(static_cast<float>(H * W));

    cufftComplex* d_input_complex = nullptr;
    cufftComplex* d_fft = nullptr;
    cufftComplex* d_ifft_input = nullptr;
    cufftComplex* d_spatial_complex = nullptr;

    float* d_low_pass = nullptr;
    float* d_high_pass = nullptr;
    float* d_phase = nullptr;
    float* d_low_attn = nullptr;
    float* d_high_attn = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input_complex, complex_bytes));
    CUDA_CHECK(cudaMalloc(&d_fft, complex_bytes));
    CUDA_CHECK(cudaMalloc(&d_ifft_input, complex_bytes));
    CUDA_CHECK(cudaMalloc(&d_spatial_complex, complex_bytes));

    CUDA_CHECK(cudaMalloc(&d_low_pass, float_bytes));
    CUDA_CHECK(cudaMalloc(&d_high_pass, float_bytes));
    CUDA_CHECK(cudaMalloc(&d_phase, float_bytes));
    CUDA_CHECK(cudaMalloc(&d_low_attn, float_bytes));
    CUDA_CHECK(cudaMalloc(&d_high_attn, float_bytes));

    real_to_complex_kernel<<<blocks_total, threads>>>(
        x,
        d_input_complex,
        total
    );
    CUDA_CHECK(cudaGetLastError());

    CUFFT_CHECK(cufftExecC2C(
        plan,
        d_input_complex,
        d_fft,
        CUFFT_FORWARD
    ));

    make_low_high_phase_shift_kernel<<<blocks_total, threads>>>(
        d_fft,
        d_low_pass,
        d_high_pass,
        d_phase,
        N, C, H, W,
        cuton,
        ortho_scale
    );
    CUDA_CHECK(cudaGetLastError());

    attention_1x1_kernel<<<blocks_total, threads>>>(
        d_low_pass,
        low_w,
        d_low_attn,
        N, C, H, W
    );
    CUDA_CHECK(cudaGetLastError());

    attention_1x1_kernel<<<blocks_total, threads>>>(
        d_high_pass,
        high_w,
        d_high_attn,
        N, C, H, W
    );
    CUDA_CHECK(cudaGetLastError());

    reconstruct_ifft_input_kernel<<<blocks_pixel, threads>>>(
        d_low_pass,
        d_high_pass,
        d_phase,
        d_low_attn,
        d_high_attn,
        d_ifft_input,
        N, C, H, W
    );
    CUDA_CHECK(cudaGetLastError());

    CUFFT_CHECK(cufftExecC2C(
        plan,
        d_ifft_input,
        d_spatial_complex,
        CUFFT_INVERSE
    ));

    complex_to_real_scale_kernel<<<blocks_total, threads>>>(
        d_spatial_complex,
        y,
        total,
        ortho_scale
    );
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_input_complex));
    CUDA_CHECK(cudaFree(d_fft));
    CUDA_CHECK(cudaFree(d_ifft_input));
    CUDA_CHECK(cudaFree(d_spatial_complex));

    CUDA_CHECK(cudaFree(d_low_pass));
    CUDA_CHECK(cudaFree(d_high_pass));
    CUDA_CHECK(cudaFree(d_phase));
    CUDA_CHECK(cudaFree(d_low_attn));
    CUDA_CHECK(cudaFree(d_high_attn));
}


struct AFAKernelCuda {
    AFAKernelCuda(const OrtApi& api, const OrtKernelInfo* info) {
        (void)api;
        (void)info;

        cuton_ = 0.1f;
        plan_ready_ = false;
        cached_NC_ = 0;
        cached_H_ = 0;
        cached_W_ = 0;
        plan_ = 0;
    }

    ~AFAKernelCuda() {
        if (plan_ready_) {
            cufftDestroy(plan_);
        }
    }

    void EnsurePlan(int64_t N, int64_t C, int64_t H, int64_t W) {
        const int64_t NC = N * C;

        if (plan_ready_ &&
            cached_NC_ == NC &&
            cached_H_ == H &&
            cached_W_ == W) {
            return;
        }

        if (plan_ready_) {
            cufftDestroy(plan_);
            plan_ready_ = false;
        }

        int n[2] = {
            static_cast<int>(H),
            static_cast<int>(W)
        };

        int inembed[2] = {
            static_cast<int>(H),
            static_cast<int>(W)
        };

        int onembed[2] = {
            static_cast<int>(H),
            static_cast<int>(W)
        };

        int istride = 1;
        int ostride = 1;
        int idist = static_cast<int>(H * W);
        int odist = static_cast<int>(H * W);

        CUFFT_CHECK(cufftPlanMany(
            &plan_,
            2,
            n,
            inembed,
            istride,
            idist,
            onembed,
            ostride,
            odist,
            CUFFT_C2C,
            static_cast<int>(NC)
        ));

        cached_NC_ = NC;
        cached_H_ = H;
        cached_W_ = W;
        plan_ready_ = true;
    }

    void Compute(OrtKernelContext* context) {
        Ort::KernelContext ctx(context);

        Ort::ConstValue input_x = ctx.GetInput(0);
        Ort::ConstValue input_low_w = ctx.GetInput(1);
        Ort::ConstValue input_high_w = ctx.GetInput(2);

        const float* x = input_x.GetTensorData<float>();
        const float* low_w = input_low_w.GetTensorData<float>();
        const float* high_w = input_high_w.GetTensorData<float>();

        std::vector<int64_t> x_shape =
            input_x.GetTensorTypeAndShapeInfo().GetShape();

        std::vector<int64_t> low_w_shape =
            input_low_w.GetTensorTypeAndShapeInfo().GetShape();

        std::vector<int64_t> high_w_shape =
            input_high_w.GetTensorTypeAndShapeInfo().GetShape();

        if (x_shape.size() != 4) {
            throw std::runtime_error("kdafa::AFA CUDA expects x shape [N, C, H, W].");
        }

        const int64_t N = x_shape[0];
        const int64_t C = x_shape[1];
        const int64_t H = x_shape[2];
        const int64_t W = x_shape[3];

        if (low_w_shape.size() != 4 || high_w_shape.size() != 4) {
            throw std::runtime_error(
                "kdafa::AFA CUDA expects attention weights shape [C, C, 1, 1]."
            );
        }

        if (low_w_shape[0] != C || low_w_shape[1] != C ||
            high_w_shape[0] != C || high_w_shape[1] != C ||
            low_w_shape[2] != 1 || low_w_shape[3] != 1 ||
            high_w_shape[2] != 1 || high_w_shape[3] != 1) {
            throw std::runtime_error(
                "kdafa::AFA CUDA attention weight shape mismatch."
            );
        }

        EnsurePlan(N, C, H, W);

        Ort::UnownedValue output = ctx.GetOutput(0, x_shape);
        float* y = output.GetTensorMutableData<float>();

        compute_afa_cuda(
            x,
            low_w,
            high_w,
            y,
            N, C, H, W,
            cuton_,
            plan_
        );
    }

    float cuton_;
    cufftHandle plan_;
    bool plan_ready_;
    int64_t cached_NC_;
    int64_t cached_H_;
    int64_t cached_W_;
};


struct AFAOpCuda : Ort::CustomOpBase<AFAOpCuda, AFAKernelCuda> {
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
        return new AFAKernelCuda(api, info);
    }

    const char* GetName() const {
        return "AFA";
    }

    const char* GetExecutionProviderType() const {
        return "CUDAExecutionProvider";
    }

    size_t GetInputTypeCount() const {
        return 3;
    }

    ONNXTensorElementDataType GetInputType(size_t index) const {
        (void)index;
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }

    size_t GetOutputTypeCount() const {
        return 1;
    }

    ONNXTensorElementDataType GetOutputType(size_t index) const {
        (void)index;
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
};


static AFAOpCuda c_AFAOpCuda;


extern "C" __declspec(dllexport) OrtStatus* ORT_API_CALL RegisterCustomOps(
    OrtSessionOptions* options,
    const OrtApiBase* api_base
) {
    const OrtApi* api = api_base->GetApi(ORT_API_VERSION);

    Ort::InitApi(api);

    OrtCustomOpDomain* domain = nullptr;

    OrtStatus* status = api->CreateCustomOpDomain("kdafa", &domain);
    if (status != nullptr) {
        return status;
    }

    status = api->CustomOpDomain_Add(domain, &c_AFAOpCuda);
    if (status != nullptr) {
        return status;
    }

    status = api->AddCustomOpDomain(options, domain);
    if (status != nullptr) {
        return status;
    }

    return nullptr;
}