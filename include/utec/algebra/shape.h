#ifndef PROG3_PF_EPIC1_FEATURE1_V2026_01_SHAPE_H
#define PROG3_PF_EPIC1_FEATURE1_V2026_01_SHAPE_H

#pragma once

#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <vector>

namespace utec::tf {

    class Shape {
    public:
        Shape() = default;
        Shape(std::initializer_list<int> dims);
        explicit Shape(const std::vector<int>& dims);

        [[nodiscard]] std::size_t rank() const noexcept;
        [[nodiscard]] std::size_t numel() const;
        [[nodiscard]] const std::vector<int>& dims() const noexcept;
        [[nodiscard]] int operator[](std::size_t i) const;

        [[nodiscard]] std::size_t size() const noexcept;
        [[nodiscard]] std::size_t total_size() const;

        friend bool operator==(const Shape&, const Shape&) = default;

    private:
        std::vector<int> dims_;
        void validate() const;
    };

    Shape::Shape(std::initializer_list<int> dims) : dims_(dims) {
        for (int d : dims_) {
            if (d <= 0) {
                throw std::invalid_argument("dim invalida");
            }
        }
    }

    Shape::Shape(const std::vector<int>& dims): dims_(dims) {
        for (int d : dims_) {
            if (d <= 0) {
                throw std::invalid_argument("dim invalida");
            }
        }
    }

    inline void Shape::validate() const {
        for (int dim : dims_) {
            if (dim <= 0) {
                throw std::invalid_argument("toda dimension debe ser positiva");
            }
        }
    }

    [[nodiscard]] std::size_t Shape::rank() const noexcept {
        return dims_.size();
    }

    [[nodiscard]] std::size_t Shape::numel() const {
        size_t dim=1;
        if (dims_.empty())
            return 0;
        for (const auto& d:dims_)
            dim *= d;
        return dim;
    }

    [[nodiscard]] const std::vector<int>& Shape::dims() const noexcept {
        return dims_;
    }

    [[nodiscard]] int Shape::operator[](std::size_t i) const {
        if (!(i >= this->rank()))
            return dims_[i];
        else
            throw std::out_of_range(std::string("index out of range"));
    }

    [[nodiscard]] std::size_t Shape::size() const noexcept {
        return rank();
    }

    [[nodiscard]] std::size_t Shape::total_size() const {
        return numel();
    }

}


#endif //PROG3_PF_EPIC1_FEATURE1_V2026_01_SHAPE_H
