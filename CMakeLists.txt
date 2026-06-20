cmake_minimum_required(VERSION 3.10.2)
project(prog3_pf_epic1_feature1_v2026_1)

set(CMAKE_CXX_STANDARD 23)
add_compile_definitions(_SILENCE_CXX23_DENORM_DEPRECATION_WARNING)

if(UNIX AND NOT APPLE)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()

if(MINGW OR CYGWIN)
    add_definitions(-O3)
endif()

# 1. Definir los archivos fuente
file(GLOB source_files
        main.cpp
        src/*.cpp)

# 2. CREAR EL EJECUTABLE (Esta es la pieza que faltaba)
# Sin esto, las líneas de abajo no saben a quién referirse
add_executable(${PROJECT_NAME} ${source_files}
        include/utec/nn/nn_ops.h
        tests/epic1/feature2_numeric_ops_test.cpp
        include/utec/nn/nn_interfaces.h
        include/utec/nn/nn_dense.h
        include/utec/nn/nn_convolution.h
        include/utec/nn/nn_pooling.h
        include/utec/nn/nn_flatten.h
        include/utec/nn/nn_activation.h
        include/utec/nn/nn_loss.h
        include/utec/nn/nn_optimizer.h
        include/utec/nn/neural_network.h
)

# 3. Configurar directorios de inclusión generales
target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen
)


# 5. Configurar TBB (solo para Linux)
if(UNIX AND NOT APPLE)
    find_package(TBB)
    if(TBB_FOUND)
        target_link_libraries(${PROJECT_NAME} PRIVATE TBB::tbb)
    endif()
endif()

add_subdirectory(autograder_epic3)