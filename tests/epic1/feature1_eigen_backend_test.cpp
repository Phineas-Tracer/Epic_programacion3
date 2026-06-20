//
// Created by jcedr on 23/04/2026.
//

#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include "utec/algebra/tensor_backend.h"
using utec::tf::Shape;
using utec::tf::Tensor;
namespace {
    void expect(bool condition, std::string_view message) {
        if (!condition) {
            throw std::runtime_error(std::string(message));
        }
    }
    template <class Exception, class Fn>
    void expect_throw(Fn&& fn, std::string_view message) {
        try {
            fn();
        } catch (const Exception&) {
            return;
        }
        throw std::runtime_error(std::string(message));
    }
} // namespace


int main() {
    // Escenario de exito: creacion y metadatos.
    auto a = Tensor<float>::zeros(Shape{2, 3});
    expect(a.rank() == 2, "rank incorrecto");
    expect(a.numel() == 6, "numel incorrecto");
    expect(a.shape() == Shape{2, 3}, "shape incorrecto");

    // Escenario de exito: acceso de lectura y escritura.
    a(1, 2) = 7.0f;
    expect(std::fabs(a(1, 2) - 7.0f) < 1e-6f, "acceso por indice fallo");

    // Escenario de exito: construccion desde datos.
    auto b = Tensor<float>::from_data(Shape{2, 3}, {1, 2, 3, 4, 5, 6});
    expect(std::fabs(b(0, 0) - 1.0f) < 1e-6f, "from_data valor 0 incorrecto");
    expect(std::fabs(b(1, 2) - 6.0f) < 1e-6f, "from_data valor final incorrecto");

    // Escenario de exito: suma elemento a elemento.
    auto c = a + Tensor<float>::ones(Shape{2, 3});
    expect(std::fabs(c(1, 2) - 8.0f) < 1e-6f, "suma incorrecta");
    expect(std::fabs(c(0, 0) - 1.0f) < 1e-6f, "suma base incorrecta");

    // Escenario de exito: reshape compatible.
    auto d = c.reshape(Shape{3, 2});
    expect(d.shape() == Shape{3, 2}, "reshape shape incorrecto");
    expect(d.numel() == 6, "reshape numel incorrecto");

    // Escenario de falla: dimension invalida.
    expect_throw<std::invalid_argument>(
    [] { (void)Tensor<float>::zeros(Shape{2, -1}); },
    "shape invalido no lanzo excepcion");

    // Escenario de falla: cantidad de datos incompatible.
    expect_throw<std::invalid_argument>(
    [] { (void)Tensor<float>::from_data(Shape{2, 3}, {1, 2, 3}); },
    "from_data debio fallar por cantidad de datos");

    // Escenario de falla: acceso fuera de rango.
    expect_throw<std::out_of_range>(
    [&] { (void)a(3, 0); },
    "acceso fuera de rango no lanzo excepcion");

    // Escenario de falla: reshape incompatible.
    expect_throw<std::invalid_argument>(
    [&] { (void)c.reshape(Shape{4, 2}); },
    "reshape incompatible no lanzo excepcion");

    // Escenario de falla: suma con shapes incompatibles.
    expect_throw<std::invalid_argument>(
    [&] {
    auto small = Tensor<float>::ones(Shape{3, 2});
    (void)(c + small);
    },"suma incompatible no lanzo excepcion");
}
