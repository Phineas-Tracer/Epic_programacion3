//
// Created by jcedr on 19/05/2026.
//

#ifndef PROG3_PF_EPIC1_FEATURE1_V2026_1_NN_OPTIMIZER_H
#define PROG3_PF_EPIC1_FEATURE1_V2026_1_NN_OPTIMIZER_H
#pragma once

#include <memory>
#include <vector>
#include <unordered_map>

#include "tensor_backend.h"
#include "nn_interfaces.h"
#include "nn_loss.h"

namespace utec::tf {
    namespace optimizers {
        struct SGD {
            float learning_rate;
            explicit SGD(float lr = 0.01f): learning_rate(lr) {}
        };
    }
    class Sequential {
    public:
        Sequential(): build_(false), compiled_(false) {}
        template <typename LayerT>
        void add(LayerT layer);
        void compile(optimizers::SGD optimizer, losses::CategoricalCrossentropy loss);
        Tensor<float> predict(const Tensor<float>& batch);
        [[nodiscard]] std::unordered_map<std::string, Tensor<float>> parameters() const;
        bool compiled() const;
    private:
        std::vector<std::shared_ptr<Layer>> layers_;
        optimizers::SGD optimizer_;
        losses::CategoricalCrossentropy loss_;
        bool build_ = false;
        bool compiled_ = false;
        Shape output_shape_;
    };

    template <typename LayerT>
    void Sequential::add(LayerT layer) {
        layers_.push_back(std::make_shared<LayerT>(std::move(layer)));
    }

    void Sequential::compile(optimizers::SGD optimizer, losses::CategoricalCrossentropy loss) {
        optimizer_ = optimizer;
        loss_ = loss;
        compiled_ = true;
    }

    Tensor<float> Sequential::predict(const Tensor<float>& batch) {
        if (layers_.empty()) {
            throw std::invalid_argument("No layers found");
        }
        if (!build_) {
            Shape current_shape = batch.shape();
            for (auto& layer : layers_) {
                layer->build(current_shape);
                current_shape = layer->output_shape();
            }
            build_ = true;
        }
        Tensor<float> x = batch;
        for (auto& layer : layers_)
            x = layer->forward(x);
        return x;
    }

    std::unordered_map<std::string, Tensor<float>> Sequential::parameters() const {
        std::unordered_map<std::string, Tensor<float>> params;
        int layer_idx = 0;
        for (const auto& layer : layers_) {
            auto layer_params = layer->parameters();
            for (auto& [name, tensor] : layer_params) {
                std::string message = "layer_" + std::to_string(layer_idx) + "/" + name;
                params.emplace(message, tensor);
            }
            ++layer_idx;
        }
        return params;
    }
    
    bool Sequential::compiled() const {
        return compiled_;
    }
}




#endif //PROG3_PF_EPIC1_FEATURE1_V2026_1_NN_OPTIMIZER_H
