//
// Created by jcedr on 19/05/2026.
//

#ifndef PROG3_PF_EPIC1_FEATURE1_V2026_1_NN_ACTIVATION_H
#define PROG3_PF_EPIC1_FEATURE1_V2026_1_NN_ACTIVATION_H

#include "utec/algebra/tensor_backend.h"

namespace utec::tf {
    enum class Activation {
        Linear,
        Relu,
        Softmax
    };

    template<typename T>
    Tensor<T> apply_activation(const Tensor<T>& x, Activation activation) {
        switch (activation) {
            case Activation::Linear:
                return x;
            case Activation::Relu: {
                Tensor<T> out = x;
                for (size_t i = 0; i < out.numel(); ++i) {
                    if (out[i] < 0)
                        out[i] = 0;
                }
                return out;
            }
            case Activation::Softmax: {
                if (x.rank() != 2)
                    throw std::invalid_argument("softmax requiere rank 2");
                Tensor<T> out = x;
                int batch = x.shape()[0];
                int classes = x.shape()[1];
                for (int b = 0; b < batch; ++b) {
                    T sum = T{};
                    for (int c = 0; c < classes; ++c)
                        sum += std::exp(x(b,c));
                    for (int c = 0; c < classes; ++c)
                        out(b,c) = std::exp(x(b,c)) / sum;
                }
                return out;
            }
            default:
                throw std::invalid_argument("activation no soportada");
        }
    }

}


#endif //PROG3_PF_EPIC1_FEATURE1_V2026_1_NN_ACTIVATION_H