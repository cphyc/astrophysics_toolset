cmake_minimum_required(VERSION 3.18)
project(astrophysics_toolset LANGUAGES C CXX Fortran)

# Required so scikit-build-core can locate python, numpy, etc.
find_package(Python COMPONENTS Interpreter Development.Module NumPy REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find Cython (installed via pip)
find_program(CYTHON_EXECUTABLE cython)
if(NOT CYTHON_EXECUTABLE)
    message(FATAL_ERROR "Cython not found")
endif()

# Find f2py (installed via pip)
find_program(F2PY_EXECUTABLE f2py REQUIRED)
if(NOT F2PY_EXECUTABLE)
    message(FATAL_ERROR "f2py not found")
endif()

# Function to build Cython modules
function(add_cython_module module_name pyx_file output_dir)
    get_filename_component(module_base ${module_name} NAME_WE)
    set(cpp_file ${CMAKE_CURRENT_BINARY_DIR}/${module_base}.cpp)

    add_custom_command(
        OUTPUT ${cpp_file}
        COMMAND ${CYTHON_EXECUTABLE} -3 --cplus -o ${cpp_file} ${pyx_file}
        DEPENDS ${pyx_file}
        COMMENT "Cythonizing ${pyx_file}"
    )

    python_add_library(${module_base} MODULE ${cpp_file})
    target_include_directories(${module_base} PRIVATE ${Python_NumPy_INCLUDE_DIRS})
    target_link_libraries(${module_base} PRIVATE OpenMP::OpenMP_CXX)
    set_target_properties(${module_base} PROPERTIES
        OUTPUT_NAME "${module_base}.${Python_SOABI}"
        SUFFIX ".so"
    )

    install(TARGETS ${module_base}
            LIBRARY DESTINATION ${output_dir}
            COMPONENT python_modules)
endfunction()

# Function to build f2py modules
function(add_f2py_module module_name f90_file output_dir)
    get_filename_component(module_base ${module_name} NAME_WE)
    get_filename_component(f90_filename ${f90_file} NAME)

    # Output .so file location
    set(module_so ${CMAKE_CURRENT_BINARY_DIR}/${module_base}.${Python_SOABI}.so)

    # Use f2py to compile everything in one go
    add_custom_command(
        OUTPUT ${module_so}
        COMMAND ${F2PY_EXECUTABLE}
                -c ${f90_file}
                -m ${module_base}
                --build-dir ${CMAKE_CURRENT_BINARY_DIR}/f2py_${module_base}
                --f90flags=-O3
        DEPENDS ${f90_file}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Building f2py module ${module_base} from ${f90_filename}"
    )

    add_custom_target(${module_base}_target ALL DEPENDS ${module_so})

    install(FILES ${module_so}
            DESTINATION ${output_dir}
            COMPONENT python_modules)
endfunction()

# -------------------------------------------------------------------
# Build Cython modules in astrophysics_toolset/ramses
# -------------------------------------------------------------------
add_cython_module(hilbert
    ${CMAKE_CURRENT_SOURCE_DIR}/astrophysics_toolset/ramses/hilbert.pyx
    astrophysics_toolset/ramses)

add_cython_module(oct_handler
    ${CMAKE_CURRENT_SOURCE_DIR}/astrophysics_toolset/ramses/oct_handler.pyx
    astrophysics_toolset/ramses)

add_cython_module(octree_converter
    ${CMAKE_CURRENT_SOURCE_DIR}/astrophysics_toolset/ramses/octree_converter.pyx
    astrophysics_toolset/ramses)

# -------------------------------------------------------------------
# Build Fortran modules in astrophysics_toolset/ramses
# -------------------------------------------------------------------
add_f2py_module(_convert_to_single_precision
    ${CMAKE_CURRENT_SOURCE_DIR}/astrophysics_toolset/ramses/_convert_to_single_precision.f90
    astrophysics_toolset/ramses)
