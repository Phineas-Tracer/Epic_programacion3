//
// Created by jcedr on 19/05/2026.
//

#ifndef PROG3_PF_EPIC1_FEATURE1_V2026_1_NN_DENSE_H
#define PROG3_PF_EPIC1_FEATURE1_V2026_1_NN_DENSE_H

#include "utec/algebra/tensor_backend.h"
#include "utec/nn/nn_activation.h"
#include "utec/nn/nn_ops.h"
#include "utec/nn/nn_interfaces.h"

namespace utec::tf {
    namespace layers {
        class Dense : public Layer {
        public:
            Dense(int units, Activation activation = Activation::Linear): units_(units), activation_(activation) {
                if (units <= 0) {
                    throw std::invalid_argument("Error");
                }
            }
            void build(const Shape& input_shape) override;
            Tensor<float> forward(const Tensor<float>& x) override;
            [[nodiscard]] Shape output_shape() const override;
            [[nodiscard]] std::unordered_map<std::string, Tensor<float>> parameters() const override;
            std::unique_ptr<Layer> clone() const override { return std::make_unique<Dense>(*this); }
        private:
            int units_;
            Activation activation_;
            Shape input_shape_;
            Shape output_shape_;
            Tensor<float> weights_;
            Tensor<float> bias_;
            bool built_;
        };

        void Dense::build(const Shape& input_shape) {
            if (input_shape.rank() == 0)
                throw std::runtime_error("Build Failure");
            input_shape_ = input_shape;
            int in_features;
            if (input_shape.rank() == 1) {
                in_features = input_shape[0];
                output_shape_ = utec::tf::Shape{units_};
            }
            else if (input_shape_.rank() == 2) {
                in_features = input_shape[1];
                output_shape_ = utec::tf::Shape{units_};
            }
            else {
                in_features = input_shape[input_shape.rank() - 1];
                output_shape_ = Shape{input_shape_[0], units_};
            }
            weights_ = Tensor<float>::random_uniform(Shape{in_features, units_}, -0.05f, 0.05f);
            bias_ = Tensor<float>::zeros(Shape{units_});
            built_ = true;
        }
        
        Tensor<float> Dense::forward(const Tensor<float>& x) {
            int batch_size = (x.shape().rank() == 1) ? 1 : x.shape()[0];
            Tensor<float> last_input_ = x;
            if (x.shape().rank() == 1) {
                last_input_ = x.reshape(Shape{1, x.shape()[0]});
            }
            Tensor<float> output = ops::matmul(last_input_, weights_);

            for (int b = 0; b < batch_size; ++b) {
                for (int u = 0; u < units_; ++u) {
                    output(b, u) += bias_(u);
                }
            }
            Tensor<float> activated_output = apply_activation(output, activation_);
            built_ = true;
            return activated_output.reshape(Shape{batch_size, units_});
        }

        Shape Dense::output_shape() const {
            return output_shape_;
        }

        std::unordered_map<std::string, Tensor<float>> Dense::parameters() const {
            std::unordered_map<std::string, Tensor<float>> parameters;
            parameters["weights"] = weights_;
            parameters["bias"] = bias_;
            return parameters;
        }
    }

}

#endif //PROG3_PF_EPIC1_FEATURE1_V2026_1_NN_DENSE_H
