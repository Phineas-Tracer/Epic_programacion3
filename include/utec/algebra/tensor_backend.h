//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_PF_EPIC1_FEATURE1_V2026_01_TENSOR_BACKEND_H
#define PROG3_PF_EPIC1_FEATURE1_V2026_01_TENSOR_BACKEND_H

#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

#include "utec/algebra/shape.h"

namespace utec::tf {

    template <class T>
    class Tensor {
    public:
        Tensor() = default;
        explicit Tensor(const Shape& shape);
        Tensor(const Shape& shape, const std::vector<T>& values);
        Tensor(const Shape& shape, const T& value): shape_(shape), data_(static_cast<Eigen::Index>(shape.numel())) {
            data_.setConstant(value);
        }

        static Tensor zeros(const Shape& shape);
        static Tensor ones(const Shape& shape);
        static Tensor from_data(const Shape& shape, const std::vector<T>& values);

        [[nodiscard]] const Shape& shape() const;
        [[nodiscard]] std::size_t rank() const noexcept;
        [[nodiscard]] std::size_t numel() const noexcept;

        template <typename... Ix>
        T& operator()(Ix... indices);

        template <typename... Ix>
        const T& operator()(Ix... indices) const;

        T& operator[](std::size_t i);
        const T& operator[](std::size_t i) const;

        Tensor reshape(const Shape& new_shape) const;
        Tensor operator+(const Tensor& other) const;
        Tensor operator-(const Tensor<T>& other) const;

        [[nodiscard]] std::size_t size() const noexcept;
        T& at(std::initializer_list<int> index);
        const T& at(std::initializer_list<int> index) const;

        static Tensor random_uniform(const Shape& shape, T min, T max);

    private:
        Shape shape_;
        Eigen::Array<T, Eigen::Dynamic, 1> data_;
        [[nodiscard]] std::size_t flat_index(std::span<const int> indices) const;
    };

    template<class T>
    std::size_t Tensor<T>::flat_index(std::span<const int> indices) const {
        if (indices.size() != shape_.rank()) {
            throw std::invalid_argument("numero de indices incompatible con el rank");
        }

        std::size_t flat=0;
        std::size_t stride=1;

        for (std::size_t i = shape_.rank(); i-- > 0;) {
            const int dim = shape_[i];
            const int idx = indices[i];
            if (idx < 0 || idx >= dim) {
                throw std::out_of_range("indice fuera de rango");
            }
            flat += static_cast<std::size_t>(idx) * stride;
            stride *= static_cast<std::size_t>(dim);
        }
        return flat;
    }

    template<class T>
    Tensor<T>::Tensor (const Shape& shape): shape_(shape), data_(static_cast<Eigen::Index>(shape.numel())) {
        data_.setZero();
    }

    template<class T>
    Tensor<T>::Tensor(const Shape &shape, const std::vector<T> &values): shape_(shape), data_(static_cast<Eigen::Index>(shape.numel())) {
        if (values.size() != shape_.numel()) {
            throw std::invalid_argument("cantidad de datos incompatible con la forma");
        }
        for (std::size_t i = 0; i < values.size(); ++i) {
            data_(static_cast<Eigen::Index>(i)) = values[i];
        }
    }

    template<class T>
    Tensor<T> Tensor<T>::zeros(const Shape &shape) {
        Tensor<T> out(shape);
        out.data_.setZero();
        return out;
    }

    template<class T>
    Tensor<T> Tensor<T>::ones(const Shape &shape) {
        Tensor<T> out(shape);
        out.data_.setOnes();
        return out;
    }

    template<class T>
    Tensor<T> Tensor<T>::from_data(const Shape &shape, const std::vector<T> &values) {
        return Tensor<T>(shape, values);
    }

    template<class T>
    Tensor<T> Tensor<T>::reshape(const Shape &new_shape) const {
        if (new_shape.numel() != numel()) {
            throw std::invalid_argument("reshape incompatible");
        }
        Tensor<T> out(new_shape);
        out.data_ = data_;
        return out;
    }

    template<class T>
    Tensor<T> Tensor<T>::operator+(const Tensor<T>& other) const {
        if (shape_ != other.shape_) {
            throw std::invalid_argument("shape incompatibles para sumar");
        }
        Tensor<T> out(shape_);
        out.data_ = data_ + other.data_;
        return out;
    }

    template<class T>
    Tensor<T> Tensor<T>::operator-(const Tensor<T>& other) const {
            if (shape_ != other.shape_)
                throw std::invalid_argument("shape incompatibles para resta");
            Tensor<T> out(shape_);
            out.data_ = data_ - other.data_;
            return out;
        }

    template <typename T>
    template <typename... Ix>
    T& Tensor<T>::operator()(Ix... indices) {
        static_assert((std::is_integral_v<Ix> && ...));
        const int idx[] = {static_cast<int>(indices)...};
        return data_(static_cast<Eigen::Index>(flat_index(std::span<const int>(idx, sizeof...(indices)))));
    }

    template<class T>
    const Shape &Tensor<T>::shape() const {
        return shape_;
    }

    template<class T>
    std::size_t Tensor<T>::rank() const noexcept {
         return shape_.rank();
    }

    template<class T>
    std::size_t Tensor<T>::numel() const noexcept {
        return shape_.numel();
    }

    template <typename T>
    template <typename... Ix>
    const T& Tensor<T>::operator()(Ix... indices) const {
        static_assert((std::is_integral_v<Ix> && ...));
        const int idx[] = {static_cast<int>(indices)...};
        return data_(static_cast<Eigen::Index>(flat_index(std::span<const int>(idx, sizeof...(indices)))));
    }

    template <class T>
    T& Tensor<T>::operator[](std::size_t i) {
        return data_(static_cast<Eigen::Index>(i));
    }

    template <class T>
    const T& Tensor<T>::operator[](std::size_t i) const {
        return data_(static_cast<Eigen::Index>(i));
    }

    template<class T>
    [[nodiscard]] std::size_t Tensor<T>::size() const noexcept {
        return shape_.numel();
    }

    template<class T>
    T& Tensor<T>::at(std::initializer_list<int> index) {
        return data_(static_cast<Eigen::Index>(flat_index(std::span<const int>(index.begin(), index.size()))));
    }

    template<class T>
    const T& Tensor<T>::at(std::initializer_list<int> index) const {
        return data_(static_cast<Eigen::Index>(flat_index(std::span<const int>(index.begin(), index.size()))));
    }

    template<class T>
    Tensor<T> Tensor<T>::random_uniform(const Shape& shape, T min, T max) {
        Tensor<T> out(shape);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dist(min, max);
        for (std::size_t i = 0; i < out.numel(); ++i)
            out[i] = dist(gen);
        return out;
    }
}







#endif //PROG3_PF_EPIC1_FEATURE1_V2026_01_TENSOR_BACKEND_H
