//
// Created by jcedr on 19/05/2026.
//

#ifndef PROG3_PF_EPIC1_FEATURE1_V2026_1_NN_POOLING_H
#define PROG3_PF_EPIC1_FEATURE1_V2026_1_NN_POOLING_H

#include "utec/algebra/tensor_backend.h"
#include "utec/nn/nn_interfaces.h"
#include "utec/nn/nn_ops.h"

namespace utec::tf {
    namespace layers {
        class MaxPooling2D : public Layer {
        public:
            explicit MaxPooling2D(std::array<int, 2> pool_size): pool_size_(pool_size) {
                if (pool_size[0] <= 0 || pool_size[1] <= 0)
                    throw std::invalid_argument("MaxPooling2D dimensions must be greater than zero.");
                strides_.rows = pool_size[0];
                strides_.cols = pool_size[1];
            }
            void build(const Shape& input_shape) override;
            Tensor<float> forward(const Tensor<float>& x) override;
            [[nodiscard]] Shape output_shape() const override;
            std::unordered_map<std::string, Tensor<float>> parameters() const override;
            std::unique_ptr<Layer> clone() const override { return std::make_unique<MaxPooling2D>(*this); }
        private:
            std::array<int, 2> pool_size_;
            //std::array<int, 2> stride_;
            Shape input_shape_;
            Shape output_shape_;
            bool built_ = false;
            Strides strides_;
            Padding padding_{Padding::Valid};
        };

        void MaxPooling2D::build(const Shape& input_shape) {
            if (input_shape.rank() != 3 && input_shape.rank() != 4)
                throw std::invalid_argument("Build Failure: MaxPooling2D requiere rango 3 o 4");
            input_shape_ = input_shape;
            bool has_batch = (input_shape.rank() == 4);
            const int batch = has_batch ? input_shape[0] : 1;
            const int in_h  = has_batch ? input_shape[1] : input_shape[0];
            const int in_w  = has_batch ? input_shape[2] : input_shape[1];
            const int in_c  = has_batch ? input_shape[3] : input_shape[2];
            const int pool_h = pool_size_[0];
            const int pool_w = pool_size_[1];
            if (pool_h <= 0 || pool_w <= 0)
                throw std::invalid_argument("Unacceptable Kernel");
            const int out_h = (in_h - pool_h) / strides_.rows + 1;
            const int out_w = (in_w - pool_w) / strides_.cols + 1;
            if (has_batch)
                output_shape_ = Shape{batch, out_h, out_w, in_c};
            else
                output_shape_ = Shape{out_h, out_w, in_c};
            built_ = true;
        }

        Tensor<float> MaxPooling2D::forward(const Tensor<float>& x) {
            if (x.rank() != 3 && x.rank() != 4)
                throw std::invalid_argument("MaxPooling2D forward requiere rango 3 o 4");
            if (!built_) {
                build(x.shape());
            }
            Tensor<float> input_fix = x;
            bool needed_reshape = false;
            if (x.shape().rank() == 3) {
                input_fix = x.reshape(Shape{1, x.shape()[0], x.shape()[1], x.shape()[2]});
                needed_reshape = true;
            }
            Tensor<float> out = ops::maxpool2d(input_fix, pool_size_, strides_);
            if (needed_reshape)
                out = out.reshape(Shape{out.shape()[1], out.shape()[2], out.shape()[3]});
            return out;
        }

        [[nodiscard]] Shape MaxPooling2D::output_shape() const {
            return output_shape_;
        }

        std::unordered_map<std::string, Tensor<float>> MaxPooling2D::parameters() const {
            return {};
        }
    }
}


#endif //PROG3_PF_EPIC1_FEATURE1_V2026_1_NN_POOLING_H;