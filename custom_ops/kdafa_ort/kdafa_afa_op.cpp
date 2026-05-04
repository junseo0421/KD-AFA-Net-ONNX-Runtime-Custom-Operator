// custom_ops/kdafa_ort/kdafa_afa_op.cpp

#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "pocketfft_hdronly.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using cfloat = std::complex<float>;

static inline int64_t numel_from_shape(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (auto v : shape) {
        n *= v;
    }
    return n;
}

static inline size_t offset4(
    int64_t n, int64_t c, int64_t h, int64_t w,
    int64_t C, int64_t H, int64_t W
) {
    return static_cast<size_t>(((n * C + c) * H + h) * W + w);
}

static inline float clamp_float(float v, float lo, float hi) {
    return std::max(lo, std::min(v, hi));
}

static void fftshift_2d_bchw(
    const std::vector<cfloat>& in,
    std::vector<cfloat>& out,
    int64_t N, int64_t C, int64_t H, int64_t W
) {
    const int64_t sh_h = H / 2;
    const int64_t sh_w = W / 2;

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t y = 0; y < H; ++y) {
                for (int64_t x = 0; x < W; ++x) {
                    int64_t src_y = (y + sh_h) % H;
                    int64_t src_x = (x + sh_w) % W;

                    out[offset4(n, c, y, x, C, H, W)] =
                        in[offset4(n, c, src_y, src_x, C, H, W)];
                }
            }
        }
    }
}

static void ifftshift_2d_bchw(
    const std::vector<cfloat>& in,
    std::vector<cfloat>& out,
    int64_t N, int64_t C, int64_t H, int64_t W
) {
    // torch.fft.ifftshift 기준:
    // even length에서는 fftshift와 동일하지만,
    // odd length까지 고려하면 ceil(H/2), ceil(W/2) 이동
    const int64_t sh_h = (H + 1) / 2;
    const int64_t sh_w = (W + 1) / 2;

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t y = 0; y < H; ++y) {
                for (int64_t x = 0; x < W; ++x) {
                    int64_t src_y = (y + sh_h) % H;
                    int64_t src_x = (x + sh_w) % W;

                    out[offset4(n, c, y, x, C, H, W)] =
                        in[offset4(n, c, src_y, src_x, C, H, W)];
                }
            }
        }
    }
}

static void pocketfft_c2c_2d_bchw(
    const std::vector<cfloat>& in,
    std::vector<cfloat>& out,
    int64_t N, int64_t C, int64_t H, int64_t W,
    bool forward,
    float scale
) {
    pocketfft::shape_t shape = {
        static_cast<size_t>(N),
        static_cast<size_t>(C),
        static_cast<size_t>(H),
        static_cast<size_t>(W)
    };

    pocketfft::stride_t stride = {
        static_cast<std::ptrdiff_t>(C * H * W * sizeof(cfloat)),
        static_cast<std::ptrdiff_t>(H * W * sizeof(cfloat)),
        static_cast<std::ptrdiff_t>(W * sizeof(cfloat)),
        static_cast<std::ptrdiff_t>(sizeof(cfloat))
    };

    pocketfft::shape_t axes = {2, 3};

    // nthreads=0이면 pocketfft가 가능한 논리 코어 수를 사용한다.
    size_t nthreads = 0;

    pocketfft::c2c<float>(
        shape,
        stride,
        stride,
        axes,
        forward,
        in.data(),
        out.data(),
        scale,
        nthreads
    );
}

static void compute_afa_cpu(
    const float* x,
    const float* low_w,
    const float* high_w,
    float* y,
    int64_t N, int64_t C, int64_t H, int64_t W,
    float cuton
) {
    const int64_t total = N * C * H * W;
    const float eps_min = 1e-6f;
    const float eps_max = 1e6f;

    // 1. real input -> complex input
    std::vector<cfloat> x_complex(static_cast<size_t>(total));
    for (int64_t i = 0; i < total; ++i) {
        x_complex[static_cast<size_t>(i)] = cfloat(x[i], 0.0f);
    }

    // PyTorch torch.fft.fft2(..., norm="ortho")
    // forward scale = 1 / sqrt(H * W)
    const float ortho_scale = 1.0f / std::sqrt(static_cast<float>(H * W));

    std::vector<cfloat> x_fft(static_cast<size_t>(total));
    pocketfft_c2c_2d_bchw(
        x_complex,
        x_fft,
        N, C, H, W,
        true,
        ortho_scale
    );

    // 2. fftshift
    std::vector<cfloat> x_shift(static_cast<size_t>(total));
    fftshift_2d_bchw(x_fft, x_shift, N, C, H, W);

    // 3. magnitude, phase
    std::vector<float> magnitude(static_cast<size_t>(total));
    std::vector<float> phase(static_cast<size_t>(total));

    for (int64_t i = 0; i < total; ++i) {
        const cfloat v = x_shift[static_cast<size_t>(i)];
        float mag = std::abs(v);
        mag = clamp_float(mag, eps_min, eps_max);

        magnitude[static_cast<size_t>(i)] = mag;
        phase[static_cast<size_t>(i)] = std::atan2(v.imag(), v.real());
    }

    // 4. low-pass / high-pass split
    std::vector<float> low_pass(static_cast<size_t>(total), 0.0f);
    std::vector<float> high_pass(static_cast<size_t>(total), 0.0f);

    const int64_t cy = H / 2;
    const int64_t cx = W / 2;
    const int64_t rh = static_cast<int64_t>(cuton * static_cast<float>(cy));
    const int64_t rw = static_cast<int64_t>(cuton * static_cast<float>(cx));

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t yy = 0; yy < H; ++yy) {
                for (int64_t xx = 0; xx < W; ++xx) {
                    const size_t idx = offset4(n, c, yy, xx, C, H, W);

                    bool is_low =
                        (yy >= cy - rh) &&
                        (yy <  cy + rh) &&
                        (xx >= cx - rw) &&
                        (xx <  cx + rw);

                    if (is_low) {
                        low_pass[idx] = magnitude[idx];
                    }

                    high_pass[idx] = magnitude[idx] - low_pass[idx];
                }
            }
        }
    }

    // 5. 1x1 conv attention
    // low_w, high_w shape: [C_out, C_in, 1, 1] = [C, C, 1, 1]
    std::vector<float> low_attn(static_cast<size_t>(total), 0.0f);
    std::vector<float> high_attn(static_cast<size_t>(total), 0.0f);

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t yy = 0; yy < H; ++yy) {
            for (int64_t xx = 0; xx < W; ++xx) {
                for (int64_t co = 0; co < C; ++co) {
                    float low_sum = 0.0f;
                    float high_sum = 0.0f;

                    for (int64_t ci = 0; ci < C; ++ci) {
                        const size_t in_idx = offset4(n, ci, yy, xx, C, H, W);
                        const size_t w_idx = static_cast<size_t>(co * C + ci);

                        low_sum += low_w[w_idx] * low_pass[in_idx];
                        high_sum += high_w[w_idx] * high_pass[in_idx];
                    }

                    const size_t out_idx = offset4(n, co, yy, xx, C, H, W);
                    low_attn[out_idx] = low_sum;
                    high_attn[out_idx] = high_sum;
                }
            }
        }
    }

    // 6. channel-wise softmax
    std::vector<float> low_softmax(static_cast<size_t>(total), 0.0f);
    std::vector<float> high_softmax(static_cast<size_t>(total), 0.0f);

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t yy = 0; yy < H; ++yy) {
            for (int64_t xx = 0; xx < W; ++xx) {
                float low_max = -std::numeric_limits<float>::infinity();
                float high_max = -std::numeric_limits<float>::infinity();

                for (int64_t c = 0; c < C; ++c) {
                    const size_t idx = offset4(n, c, yy, xx, C, H, W);
                    low_max = std::max(low_max, low_attn[idx]);
                    high_max = std::max(high_max, high_attn[idx]);
                }

                float low_sum_exp = 0.0f;
                float high_sum_exp = 0.0f;

                for (int64_t c = 0; c < C; ++c) {
                    const size_t idx = offset4(n, c, yy, xx, C, H, W);

                    low_softmax[idx] = std::exp(low_attn[idx] - low_max);
                    high_softmax[idx] = std::exp(high_attn[idx] - high_max);

                    low_sum_exp += low_softmax[idx];
                    high_sum_exp += high_softmax[idx];
                }

                for (int64_t c = 0; c < C; ++c) {
                    const size_t idx = offset4(n, c, yy, xx, C, H, W);

                    low_softmax[idx] /= low_sum_exp;
                    high_softmax[idx] /= high_sum_exp;
                }
            }
        }
    }

    // 7. magnitude reconstruction
    std::vector<float> mag_out(static_cast<size_t>(total), 0.0f);

    for (int64_t i = 0; i < total; ++i) {
        float v =
            low_softmax[static_cast<size_t>(i)] * low_pass[static_cast<size_t>(i)] +
            high_softmax[static_cast<size_t>(i)] * high_pass[static_cast<size_t>(i)];

        mag_out[static_cast<size_t>(i)] = clamp_float(v, eps_min, eps_max);
    }

    // 8. real/imag reconstruction using original phase
    std::vector<cfloat> fre_shift_out(static_cast<size_t>(total));

    for (int64_t i = 0; i < total; ++i) {
        float mag = mag_out[static_cast<size_t>(i)];
        float ph = phase[static_cast<size_t>(i)];

        fre_shift_out[static_cast<size_t>(i)] =
            cfloat(mag * std::cos(ph), mag * std::sin(ph));
    }

    // 9. ifftshift
    std::vector<cfloat> fre_out(static_cast<size_t>(total));
    ifftshift_2d_bchw(fre_shift_out, fre_out, N, C, H, W);

    // 10. ifft2(..., norm="ortho")
    std::vector<cfloat> spatial_complex(static_cast<size_t>(total));
    pocketfft_c2c_2d_bchw(
        fre_out,
        spatial_complex,
        N, C, H, W,
        false,
        ortho_scale
    );

    // 11. real output
    for (int64_t i = 0; i < total; ++i) {
        y[i] = spatial_complex[static_cast<size_t>(i)].real();
    }
}


struct AFAKernel {
    AFAKernel(const OrtApi& api, const OrtKernelInfo* info) {
        (void)api;
        (void)info;

        // 현재 PyTorch symbolic에서 cuton_f=0.1로 넣었지만,
        // 여기서는 우선 고정값으로 사용.
        cuton_ = 0.1f;
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
            throw std::runtime_error("kdafa::AFA expects x shape [N, C, H, W].");
        }

        const int64_t N = x_shape[0];
        const int64_t C = x_shape[1];
        const int64_t H = x_shape[2];
        const int64_t W = x_shape[3];

        if (low_w_shape.size() != 4 || high_w_shape.size() != 4) {
            throw std::runtime_error(
                "kdafa::AFA expects attention weights shape [C, C, 1, 1]."
            );
        }

        if (low_w_shape[0] != C || low_w_shape[1] != C ||
            high_w_shape[0] != C || high_w_shape[1] != C ||
            low_w_shape[2] != 1 || low_w_shape[3] != 1 ||
            high_w_shape[2] != 1 || high_w_shape[3] != 1) {
            throw std::runtime_error(
                "kdafa::AFA attention weight shape mismatch."
            );
        }

        Ort::UnownedValue output = ctx.GetOutput(0, x_shape);
        float* y = output.GetTensorMutableData<float>();

        compute_afa_cpu(
            x,
            low_w,
            high_w,
            y,
            N, C, H, W,
            cuton_
        );
    }

    float cuton_;
};


struct AFAOp : Ort::CustomOpBase<AFAOp, AFAKernel> {
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
        return new AFAKernel(api, info);
    }

    const char* GetName() const {
        return "AFA";
    }

    const char* GetExecutionProviderType() const {
        // CPU custom op
        return nullptr;
    }

    size_t GetInputTypeCount() const {
        return 3;
    }

    ONNXTensorElementDataType GetInputType(size_t index) const {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }

    size_t GetOutputTypeCount() const {
        return 1;
    }

    ONNXTensorElementDataType GetOutputType(size_t index) const {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
};


static AFAOp c_AFAOp;


extern "C" __declspec(dllexport) OrtStatus* ORT_API_CALL RegisterCustomOps(
    OrtSessionOptions* options,
    const OrtApiBase* api_base
) {
    const OrtApi* api = api_base->GetApi(ORT_API_VERSION);

    // ORT_API_MANUAL_INIT를 사용한 경우,
    // ONNX Runtime C++ wrapper가 사용할 API pointer를 초기화해야 함
    Ort::InitApi(api);

    OrtCustomOpDomain* domain = nullptr;

    // PyTorch symbolic에서 g.op("kdafa::AFA", ...)로 export했으므로
    // domain 이름은 반드시 "kdafa"여야 함
    OrtStatus* status = api->CreateCustomOpDomain("kdafa", &domain);
    if (status != nullptr) {
        return status;
    }

    // kdafa domain 안에 AFA custom op 등록
    status = api->CustomOpDomain_Add(domain, &c_AFAOp);
    if (status != nullptr) {
        return status;
    }

    // SessionOptions에 custom op domain 추가
    status = api->AddCustomOpDomain(options, domain);
    if (status != nullptr) {
        return status;
    }

    return nullptr;
}