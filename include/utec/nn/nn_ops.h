//
// Created by jcedr on 30/04/2026.
//

#ifndef PROG3_PF_EPIC1_FEATURE1_V2026_1_NN_OPS_H
#define PROG3_PF_EPIC1_FEATURE1_V2026_1_NN_OPS_H

#pragma once
#include "utec/algebra/tensor_backend.h"
#include <omp.h>

namespace utec::tf {
    struct Strides {
        int rows;
        int cols;
    };
    enum class Padding {
        Valid,
    };

    namespace ops {
        //Multiplicacion Matricial Regular
        template <typename T>
        Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b) {
            if (a.rank() != 2 || b.rank() != 2) {
                throw std::invalid_argument("matmul requiere tensores de rank 2");
            }
            const int m = a.shape()[0];
            const int k = a.shape()[1];
            const int k2 = b.shape()[0];
            const int n = b.shape()[1];
            if (k != k2) {
                throw std::invalid_argument("matmul recibio dimensiones incompatibles");
            }
            Tensor<T> out = Tensor<T>::zeros(Shape{m, n});
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    T acc = T{};
                    for (int t = 0; t < k; ++t)
                        acc += a(i, t) * b(t, j);
                    out(i, j) = acc;
                }
            }
            return out;
        }

        //Multiplicacion Matricial Handaman
        template <typename T>
        Tensor<T> multiply(const Tensor<T>& a, const Tensor<T>& b) {
            if (a.shape() != b.shape())
                throw std::invalid_argument("multiply recibio dimensiones incompatibles");
            Tensor<T> out(a.shape());
            for (int i = 0; i < a.shape()[0]; ++i)
                for (int j = 0; j < a.shape()[1]; ++j)
                    out(i,j) = a(i,j) * b(i,j);
            return out;
        }

        //Trasponer una Matriz
        template <typename T>
        Tensor<T> transpose2d(const Tensor<T>& x) {
            if (x.rank() != 2)
                throw std::invalid_argument("transpose2d requiere una matriz de rank 2");
            int n = x.shape()[0];
            int m = x.shape()[1];
            Tensor<T> out(Shape{m, n});
            for (std::size_t i=0; i<m ;i++)
                for (std::size_t j=0; j<n; ++j)
                    out(i,j) = x(j,i);
            return out;
        }

        template <typename T>
        Tensor<T> flatten_batch(const Tensor<T>& x) {
            if (x.rank() < 2)
                throw std::invalid_argument("flatten_batch requiere rank >= 2");
            std::size_t batch = x.shape()[0];
            std::size_t features = 1;
            for (std::size_t i = 1; i < x.rank(); ++i)
                features *= x.shape()[i];
            return x.reshape(Shape{static_cast<int>(batch), static_cast<int>(features)});
        }

        template <typename T>
        Tensor<T> conv2d(const Tensor<T>& input, const Tensor<T>& kernel, Strides strides = {1,1}, Padding padding = Padding::Valid) {
            if (padding != Padding::Valid) {
                throw std::invalid_argument("solo se soporta Padding::Valid en este feature");
            }
            if (input.rank() != 4 || kernel.rank() != 4) {
                throw std::invalid_argument("conv2d requiere input y kernel de rank 4");
            }
            if (strides.rows <= 0 || strides.cols <= 0) {
                throw std::invalid_argument("stride invalido");
            }
            const int batch = input.shape()[0];
            const int in_h = input.shape()[1];
            const int in_w = input.shape()[2];
            const int in_c = input.shape()[3];
            const int k_h = kernel.shape()[0];
            const int k_w = kernel.shape()[1];
            const int k_c = kernel.shape()[2];
            const int out_c = kernel.shape()[3];
            if (in_c != k_c) {
                throw std::invalid_argument("conv2d recibio canales incompatibles");
            }
            if (k_h > in_h || k_w > in_w) {
                throw std::invalid_argument("kernel mas grande que la entrada");
            }
            const int out_h = (in_h - k_h) / strides.rows + 1;
            const int out_w = (in_w - k_w) / strides.cols + 1;
            Tensor<T> out = Tensor<T>::zeros(Shape{batch, out_h, out_w, out_c});
            #pragma omp parallel for collapse(4)
            for (int n = 0; n < batch; ++n) {
                for (int oh = 0; oh < out_h; ++oh) {
                    for (int ow = 0; ow < out_w; ++ow) {
                        for (int oc = 0; oc < out_c; ++oc) {
                            T acc = T{};
                            for (int kh = 0; kh < k_h; ++kh) {
                                for (int kw = 0; kw < k_w; ++kw) {
                                    for (int ic = 0; ic < in_c; ++ic) {
                                        const int ih = oh * strides.rows + kh;
                                        const int iw = ow * strides.cols + kw;
                                        acc += input(n, ih, iw, ic) * kernel(kh, kw, ic, oc);
                                    }
                                }
                            }
                            out(n, oh, ow, oc) = acc;
                        }
                    }
                }
            }
            return out;
        }

        template <typename T>
        Tensor<T> maxpool2d(const Tensor<T>& input, std::array<int,2> pool_size, Strides strides) {
            if (input.rank() != 4) {
                throw std::invalid_argument("maxpool2d requiere input rank 4");
            }
            if (pool_size[0] <= 0 || pool_size[1] <= 0) {
                throw std::invalid_argument("pool_size invalido");
            }
            if (strides.rows <= 0 || strides.cols <= 0) {
                throw std::invalid_argument("stride invalido");
            }
            const int batch = input.shape()[0];
            const int in_h  = input.shape()[1];
            const int in_w  = input.shape()[2];
            const int in_c  = input.shape()[3];
            const int pool_h = pool_size[0];
            const int pool_w = pool_size[1];
            if (pool_h > in_h || pool_w > in_w) {
                throw std::invalid_argument("pool_size mas grande que input");
            }
            const int out_h = (in_h - pool_h) / strides.rows + 1;
            const int out_w = (in_w - pool_w) / strides.cols + 1;
            Tensor<T> out = Tensor<T>::zeros(Shape{batch, out_h, out_w, in_c});
            #pragma omp parallel for collapse(4)
            for (int n = 0; n < batch; ++n) {
                for (int oh = 0; oh < out_h; ++oh) {
                    for (int ow = 0; ow < out_w; ++ow) {
                        for (int c = 0; c < in_c; ++c) {
                            T mx = input(n, oh * strides.rows, ow * strides.cols, c);
                            for (int ph = 0; ph < pool_h; ++ph) {
                                for (int pw = 0; pw < pool_w; ++pw) {
                                    const int ih = oh * strides.rows + ph;
                                    const int iw = ow * strides.cols + pw;
                                    if (input(n, ih, iw, c) > mx) {
                                        mx = input(n, ih, iw, c);
                                    }
                                }
                            }
                            out(n, oh, ow, c) = mx;
                        }
                    }
                }
            }
            return out;
        }
    }
}

#endif //PROG3_PF_EPIC1_FEATURE1_V2026_1_NN_OPS_H
