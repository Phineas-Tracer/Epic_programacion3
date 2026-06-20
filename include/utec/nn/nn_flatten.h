//
// Created by jcedr on 19/05/2026.
//

#ifndef PROG3_PF_EPIC1_FEATURE1_V2026_1_NN_FLATTEN_H
#define PROG3_PF_EPIC1_FEATURE1_V2026_1_NN_FLATTEN_H

#include "utec/algebra/tensor_backend.h"
#include "utec/nn/nn_interfaces.h"
#include "utec/nn/nn_ops.h"


namespace utec::tf {
    namespace layers {
        class Flatten : public Layer {
        public:
            void build(const Shape& input_shape) override {
                if (input_shape.rank() < 2)
                    throw std::invalid_argument("Flatten requiere rank >= 2");
                input_shape_ = input_shape;
                int batch = input_shape[0];
                int features = 1;
                for (size_t i = 1; i < input_shape.rank(); ++i)
                    features *= input_shape[i];
                output_shape_ = Shape{static_cast<int>(input_shape.numel())};
                built_ = true;
            }

            Tensor<float> forward(const Tensor<float>& x) override {
                if (!built_) {
                    throw std::invalid_argument("Flatten not built");
                }
                return ops::flatten_batch(x);
            }
            Shape output_shape() const override {
                return output_shape_;
            }
            std::unique_ptr<Layer> clone() const override {
                return std::make_unique<Flatten>(*this);
            }
        private:
            Shape input_shape_;
            Shape output_shape_;
            bool built_ = false;
        };
    }
}

#endif //PROG3_PF_EPIC1_FEATURE1_V2026_1_NN_FLATTEN_H
