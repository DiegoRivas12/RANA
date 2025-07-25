#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <stdexcept>

// ==================== Estructuras ====================
struct Point {
    double x, y, z;
};

struct Tetrahedron {
    Point p1, p2, p3, p4;
};

// ==================== Variables globales ====================
GLuint VAO_points, VBO_points;
GLuint VAO_lines, VBO_lines;
GLuint shaderProgramPoints, shaderProgramLines;

float yaw = -90.0f, pitch = 0.0f;
float sensitivity = 0.2f;
float radius = 5.0f;
bool firstMouse = true;
double lastX = 400, lastY = 300;
float fov = 45.0f;

glm::vec3 cameraPos;
glm::vec3 cameraUp(0.0f, 1.0f, 0.0f);

// ==================== Callbacks ====================
void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // invertido
    lastX = xpos;
    lastY = ypos;

    xoffset *= sensitivity;
    yoffset *= sensitivity;

    yaw += xoffset;
    pitch += yoffset;

    if (pitch > 89.0f) pitch = 89.0f;
    if (pitch < -89.0f) pitch = -89.0f;
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    radius -= yoffset;
    if (radius < 1.0f) radius = 1.0f;
    if (radius > 50.0f) radius = 50.0f;
}

// ==================== Shaders ====================
const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
)";

const char* fragmentShaderSourcePoints = R"(
#version 330 core
out vec4 FragColor;
void main() {
    FragColor = vec4(1.0, 0.0, 0.0, 1.0); // Rojo
}
)";

const char* fragmentShaderSourceLines = R"(
#version 330 core
out vec4 FragColor;
void main() {
    FragColor = vec4(0.0, 1.0, 0.0, 1.0); // Verde
}
)";

// ==================== Funciones ====================
GLuint compileShader(const char* vertexSrc, const char* fragmentSrc) {
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexSrc, NULL);
    glCompileShader(vertexShader);

    GLint success;
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        glGetShaderInfoLog(vertexShader, 512, NULL, log);
        throw std::runtime_error(std::string("Error en vertex shader: ") + log);
    }

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentSrc, NULL);
    glCompileShader(fragmentShader);
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        glGetShaderInfoLog(fragmentShader, 512, NULL, log);
        throw std::runtime_error(std::string("Error en fragment shader: ") + log);
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return program;
}

// Lectura del modelo
void loadModel(const std::string& fileName, std::vector<Point>& points, std::vector<Tetrahedron>& tets) {
    std::ifstream in(fileName, std::ios::binary);
    if (!in) throw std::runtime_error("No se pudo abrir el archivo: " + fileName);

    size_t nPoints, nTets;
    in.read(reinterpret_cast<char*>(&nPoints), sizeof(size_t));
    points.resize(nPoints);
    in.read(reinterpret_cast<char*>(points.data()), nPoints * sizeof(Point));

    in.read(reinterpret_cast<char*>(&nTets), sizeof(size_t));
    tets.resize(nTets);
    in.read(reinterpret_cast<char*>(tets.data()), nTets * sizeof(Tetrahedron));

    std::cout << "Modelo cargado: " << nPoints << " puntos, " << nTets << " tetraedros.\n";
}

// Cargar datos en OpenGL
void updateBuffers(const std::vector<Point>& points, const std::vector<Tetrahedron>& tets) {
    std::vector<float> vertices;
    for (const auto& p : points) {
        vertices.push_back(p.x);
        vertices.push_back(p.y);
        vertices.push_back(p.z);
    }

    std::vector<float> lineVertices;
    for (const auto& t : tets) {
        auto addEdge = [&](const Point& a, const Point& b) {
            lineVertices.push_back(a.x);
            lineVertices.push_back(a.y);
            lineVertices.push_back(a.z);
            lineVertices.push_back(b.x);
            lineVertices.push_back(b.y);
            lineVertices.push_back(b.z);
        };
        addEdge(t.p1, t.p2);
        addEdge(t.p2, t.p3);
        addEdge(t.p3, t.p1);
        addEdge(t.p1, t.p4);
        addEdge(t.p2, t.p4);
        addEdge(t.p3, t.p4);
    }

    // Buffers puntos
    glGenVertexArrays(1, &VAO_points);
    glGenBuffers(1, &VBO_points);
    glBindVertexArray(VAO_points);
    glBindBuffer(GL_ARRAY_BUFFER, VBO_points);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Buffers líneas
    glGenVertexArrays(1, &VAO_lines);
    glGenBuffers(1, &VBO_lines);
    glBindVertexArray(VAO_lines);
    glBindBuffer(GL_ARRAY_BUFFER, VBO_lines);
    glBufferData(GL_ARRAY_BUFFER, lineVertices.size() * sizeof(float), lineVertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
}

// Render loop
void renderLoop(GLFWwindow* window, size_t pointCount, size_t lineCount) {
    while (!glfwWindowShouldClose(window)) {
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Actualizar posición de la cámara (orbital)
        cameraPos.x = radius * cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        cameraPos.y = radius * sin(glm::radians(pitch));
        cameraPos.z = radius * sin(glm::radians(yaw)) * cos(glm::radians(pitch));

        glm::mat4 model = glm::mat4(1.0f);
        glm::mat4 view = glm::lookAt(cameraPos, glm::vec3(0.0f), cameraUp);
        glm::mat4 projection = glm::perspective(glm::radians(fov), 800.0f / 600.0f, 0.1f, 100.0f);

        // Puntos
        glUseProgram(shaderProgramPoints);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgramPoints, "model"), 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgramPoints, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgramPoints, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

        glBindVertexArray(VAO_points);
        glPointSize(4.0f);
        glDrawArrays(GL_POINTS, 0, pointCount);

        // Líneas
        glUseProgram(shaderProgramLines);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgramLines, "model"), 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgramLines, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgramLines, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

        glBindVertexArray(VAO_lines);
        glLineWidth(1.5f);
        glDrawArrays(GL_LINES, 0, lineCount);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}

// ==================== Main ====================
int main() {
    try {
        std::vector<Point> points;
        std::vector<Tetrahedron> tets;
        loadModel("output/puntos_tiff_lungMasks_grupo2.txt.bin", points, tets);
        //loadModel("output/puntos_tiff_eyeMasks_grupo2.txt.bin", points, tets);

        if (!glfwInit()) return -1;
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        GLFWwindow* window = glfwCreateWindow(800, 600, "Visualizador Delaunay 3D", NULL, NULL);
        if (!window) { glfwTerminate(); return -1; }
        glfwMakeContextCurrent(window);

        glfwSetCursorPosCallback(window, mouse_callback);
        glfwSetScrollCallback(window, scroll_callback);
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

        gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
        glEnable(GL_DEPTH_TEST);

        shaderProgramPoints = compileShader(vertexShaderSource, fragmentShaderSourcePoints);
        shaderProgramLines = compileShader(vertexShaderSource, fragmentShaderSourceLines);

        updateBuffers(points, tets);
        renderLoop(window, points.size(), tets.size() * 12);

        glfwTerminate();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
    return 0;
}
