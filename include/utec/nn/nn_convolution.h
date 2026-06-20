//
// Created by jcedr on 19/05/2026.
//

#ifndef PROG3_PF_EPIC1_FEATURE1_V2026_1_NN_CONVOLUTION_H
#define PROG3_PF_EPIC1_FEATURE1_V2026_1_NN_CONVOLUTION_H

#include "utec/algebra/tensor_backend.h"
#include "utec/nn/nn_activation.h"
#include "utec/nn/nn_interfaces.h"
#include "nn_ops.h"

namespace utec::tf {
    namespace layers {
        class Conv2D : public Layer {
        public:
            Conv2D(int filters, std::array<int, 2> kernel_size, Activation activation): filters_(filters), kernel_size_(kernel_size), activation_(activation) {
                if (filters <= 0)
                    throw std::invalid_argument("filters must be positive");
                if (kernel_size_[0] <=0 || kernel_size_[1] <=0)
                    throw std::invalid_argument("kernel size invalid");
            }
            void build(const Shape& input_shape) override;
            Tensor<float> forward(const Tensor<float>& x) override;
            [[nodiscard]] Shape output_shape() const override;
            [[nodiscard]] std::unordered_map<std::string, Tensor<float>> parameters() const override;
            std::unique_ptr<Layer> clone() const override { return std::make_unique<Conv2D>(*this); }
        private:
            int filters_;
            std::array<int, 2> kernel_size_;
            Activation activation_;
            Shape input_shape_;
            Shape output_shape_;
            Tensor<float> weights_;
            Tensor<float> bias_;
            bool built_ = false;
            Strides strides_{1,1};
            Padding padding_{Padding::Valid};
            std::unordered_map<std::string, Tensor<float>> params_map_;
        };

        void Conv2D::build(const Shape& input_shape) {
            if (input_shape.rank() != 3 && input_shape.rank() != 4)
                throw std::invalid_argument("Build Failure: Conv2D requiere rango 3 o 4");
            input_shape_ = input_shape;
            bool has_batch = (input_shape.rank() == 4);
            const int batch = has_batch ? input_shape[0] : 1;
            const int in_h  = has_batch ? input_shape[1] : input_shape[0];
            const int in_w  = has_batch ? input_shape[2] : input_shape[1];
            const int in_c  = has_batch ? input_shape[3] : input_shape[2];
            const int k_h = kernel_size_[0];
            const int k_w = kernel_size_[1];
            weights_ = Tensor<float>::random_uniform(Shape{k_h, k_w, in_c, filters_}, -0.05f, 0.05f);
            bias_ = Tensor<float>::zeros(Shape{filters_});
            const int out_h = (in_h - k_h) / strides_.rows + 1;
            const int out_w = (in_w - k_w) / strides_.cols + 1;
            if (has_batch)
                output_shape_ = Shape{batch, out_h, out_w, filters_};
            else
                output_shape_ = Shape{out_h, out_w, filters_};
            built_ = true;
        }

        Tensor<float> Conv2D::forward(const Tensor<float>& x) {
            if (!params_map_.empty()) {
                weights_ = params_map_["weights"];
                bias_ = params_map_["bias"];
            }
            Tensor<float> input_fix = x;
            bool needed_reshape = false;
            if (x.shape().rank() == 3) {
                input_fix = x.reshape(Shape{1, x.shape()[0], x.shape()[1], x.shape()[2]});
                needed_reshape = true;
            }
            else if (x.shape().rank() != 4)
                throw std::invalid_argument("Conv2D forward requiere un tensor de rango 3 o 4");

            Tensor<float> out = ops::conv2d(input_fix, weights_, strides_, padding_);
            int batches  = out.shape()[0];
            int out_h    = out.shape()[1];
            int out_w    = out.shape()[2];
            int out_c    = out.shape()[3];
            for (int n = 0; n < batches; ++n)
                for (int oh = 0; oh < out_h; ++oh)
                    for (int ow = 0; ow < out_w; ++ow)
                        for (int oc = 0; oc < out_c; ++oc)
                            out(n, oh, ow, oc) += bias_(oc);
            if (needed_reshape)
                out = out.reshape(Shape{out.shape()[1], out.shape()[2], out.shape()[3]});
            return apply_activation(out, activation_);
        }

        [[nodiscard]] Shape Conv2D::output_shape() const {
            return output_shape_;
        }

        [[nodiscard]] std::unordered_map<std::string, Tensor<float>> Conv2D::parameters() const {
            std::unordered_map<std::string, Tensor<float>> parameters;
            parameters.emplace("weights", weights_);
            parameters.emplace("bias", bias_);
            return parameters;
        }
    }
}

#endif //PROG3_PF_EPIC1_FEATURE1_V2026_1_NN_CONVOLUTION_H
