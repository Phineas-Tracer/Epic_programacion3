//
// Created by jcedr on 19/05/2026.
//

#ifndef PROG3_PF_EPIC1_FEATURE1_V2026_1_NN_LOSS_H
#define PROG3_PF_EPIC1_FEATURE1_V2026_1_NN_LOSS_H

#include "utec/algebra/tensor_backend.h"
#include <cmath>

namespace utec::tf::losses {
    class CategoricalCrossentropy {
    public:
        float operator()(const Tensor<float>& y_true, const Tensor<float>& y_pred) const;
    };
    float CategoricalCrossentropy::operator()(const Tensor<float>& y_true, const Tensor<float>& y_pred) const {
        if (y_true.shape() != y_pred.shape()) {
            throw std::invalid_argument("Las dimensiones de y_true e y_pred deben ser iguales.");
        }
        float loss = 0.0f;
        const float epsilon = 1e-7f;
        for (std::size_t i = 0; i < y_true.size(); ++i)
            loss -= y_true[i] * std::log(y_pred[i] + epsilon);
        std::size_t batch_size = (y_true.rank() > 0) ? static_cast<std::size_t>(y_true.shape()[0]) : 1;
        return loss/static_cast<float>(batch_size);
    }
}

#endif //PROG3_PF_EPIC1_FEATURE1_V2026_1_NN_LOSS_H