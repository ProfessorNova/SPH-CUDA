cmake_minimum_required(VERSION 3.30)
project(SPH_CUDA CUDA)

set(CMAKE_CUDA_STANDARD 23)

include(FetchContent)

# Function to include a dependency via FetchContent.
function(include_dependency libName gitURL gitTag)
    FetchContent_Declare(${libName}
            GIT_REPOSITORY ${gitURL}
            GIT_TAG ${gitTag}
            GIT_SHALLOW TRUE
            GIT_PROGRESS TRUE
    )
    FetchContent_MakeAvailable(${libName})
endfunction()

set(LIB1 raylib)
find_package(${LIB1} QUIET)
if (NOT ${LIB1}_FOUND)
    message(STATUS "Getting ${LIB1} from Github")
    include_dependency(${LIB1} https://github.com/raysan5/raylib.git 5.5)
else ()
    message(STATUS "Using local ${LIB1}")
endif ()

# Add OpenGL package, required for CUDA-OpenGL interoperability.
find_package(OpenGL REQUIRED)

add_executable(SPH_CUDA main.cu)

# Specify the include directory where raylib.h is located.
# Adjust this path if raylib.h is not in the 'src' folder.
target_include_directories(SPH_CUDA PRIVATE ${raylib_SOURCE_DIR}/src)

# Link the raylib and OpenGL libraries.
target_link_libraries(SPH_CUDA PRIVATE raylib ${OPENGL_gl_LIBRARY})

# Set CUDA-specific properties.
set_target_properties(SPH_CUDA PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
