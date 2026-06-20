//
// Created by jcedr on 19/05/2026.
//

#ifndef PROG3_PF_EPIC1_FEATURE1_V2026_1_NN_INTERFACES_H
#define PROG3_PF_EPIC1_FEATURE1_V2026_1_NN_INTERFACES_H

#pragma once
#include <unordered_map>
#include <memory>
#include "utec/algebra/tensor_backend.h"
#include <string>

namespace utec::tf {
    class Layer {
    public:
        virtual ~Layer() = default;
        virtual void build(const Shape& input_shape) = 0;
        virtual Tensor<float> forward(const Tensor<float>& x) = 0;
        virtual Shape output_shape() const = 0;
        virtual std::unordered_map<std::string, Tensor<float>>parameters() const {
            return {};
        }
        virtual std::unique_ptr<Layer> clone() const = 0;
    };

    namespace layers {
        class Input : public Layer {
        public:
            explicit Input(const Shape& shape): shape_(shape) {
                if (shape.rank() == 0)
                    throw std::invalid_argument("Input shape cannot be empty");
            }
            void build(const Shape& input_shape) override { output_shape_ = input_shape; }
            Tensor<float> forward(const Tensor<float>& x) override { return x; }
            Shape output_shape() const override { return shape_; }
            std::unordered_map<std::string, Tensor<float>> parameters() const override { return {}; }
            std::unique_ptr<Layer> clone() const override { return std::make_unique<Input>(*this); }
        private:
            Shape shape_;
            Shape output_shape_;
        };
    }
}

#endif //PROG3_PF_EPIC1_FEATURE1_V2026_1_NN_INTERFACES_H

