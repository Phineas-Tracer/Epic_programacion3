//
// Created by jcedr on 19/05/2026.
//

#ifndef PROG3_PF_EPIC1_FEATURE1_V2026_1_NEURAL_NETWORK_H
#define PROG3_PF_EPIC1_FEATURE1_V2026_1_NEURAL_NETWORK_H

#pragma once
#include <memory>
#include <unordered_map>
#include <string>

#include "utec/algebra/shape.h"
#include "utec/algebra/tensor_backend.h"

#include "utec/nn/nn_ops.h"
#include "utec/nn/nn_activation.h"
#include "utec/nn/nn_convolution.h"
#include "utec/nn/nn_dense.h"
#include "utec/nn/nn_flatten.h"
#include "utec/nn/nn_interfaces.h"
#include "utec/nn/nn_loss.h"
#include "utec/nn/nn_optimizer.h"
#include "utec/nn/nn_pooling.h"


namespace utec::tf {
    class NeuralNetwork {
    public:
        NeuralNetwork() = default;
        void compile(optimizers::SGD optimizer, losses::CategoricalCrossentropy loss);
        template <typename LayerT>
        void add(LayerT layer);
        Tensor<float> predict(const Tensor<float>& batch);
        [[nodiscard]] std::unordered_map<std::string, Tensor<float>> parameters() const;
    private:
        Sequential model_;
    };

    template <typename LayerT>
    void NeuralNetwork::add(LayerT layer) {
        model_.add(std::move(layer));
    }

    void NeuralNetwork::compile(optimizers::SGD optimizer, losses::CategoricalCrossentropy loss) {
        model_.compile(optimizer, loss);
    }

    Tensor<float> NeuralNetwork::predict(const Tensor<float>& batch) {
        return model_.predict(batch);
    }

    std::unordered_map<std::string,Tensor<float>> NeuralNetwork::parameters() const {
        return model_.parameters();
    }

}



#endif //PROG3_PF_EPIC1_FEATURE1_V2026_1_NEURAL_NETWORK_H
