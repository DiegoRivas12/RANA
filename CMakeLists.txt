cmake_minimum_required(VERSION 3.10)
# Enable both C and C++ languages
project(OpenGL_OpenCV_Integration LANGUAGES C CXX)

# Estándar C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ✅ Habilitar OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    message(STATUS "OpenMP encontrado: Versión ${OpenMP_VERSION}")
    add_definitions(-DOPENMP_PARALLEL)
endif()

# ✅ Configuración de OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV encontrado: ${OpenCV_VERSION}")
else()
    message(FATAL_ERROR "OpenCV no encontrado")
endif()

# ✅ Archivos fuente
set(SOURCES
    output/viTeta.cpp
    C:/glad/src/glad.c  # Include GLAD source directly
)

# ✅ Crear ejecutable (ONCE)
add_executable(OpenGL_OpenCV_Integration ${SOURCES})

# ✅ Incluir directorios
target_include_directories(OpenGL_OpenCV_Integration PRIVATE
    C:/glad/include
    C:/glfw-3.4.bin.WIN64/include
    C:/opt/glm
    ${OpenCV_INCLUDE_DIRS}
)

# ✅ Directorios de librerías
link_directories(
    C:/glfw-3.4.bin.WIN64/lib-mingw-w64
)

# ✅ Enlazar librerías principales
target_link_libraries(OpenGL_OpenCV_Integration PRIVATE
    glfw3
    opengl32
    gdi32
    ${OpenCV_LIBS}
)

# ✅ Si OpenMP está disponible, enlazarlo
if(OpenMP_FOUND)
    target_link_libraries(OpenGL_OpenCV_Integration PRIVATE OpenMP::OpenMP_CXX)
endif()

# ✅ Configuración para Windows
if(WIN32)
    add_definitions(-DGLFW_INCLUDE_NONE)
    # Force console subsystem if needed
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-subsystem,console")#para usar main consola
endif()