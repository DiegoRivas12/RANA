#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <sstream>
#include <map>
#include <random>

// ==================== Estructuras ====================
struct Point { double x, y, z; };
struct Tetrahedron { Point p1, p2, p3, p4; };

struct ModelPart {
    GLuint VAO_points, VBO_points;
    GLuint VAO_lines, VBO_lines;
    size_t pointCount, lineCount;
    glm::vec3 color;
};

struct Range {
    double minX, maxX, minY, maxY, minZ, maxZ;
};

// ==================== Variables globales ====================
std::vector<ModelPart> modelParts;
std::map<std::string, Range> rangos;

float yaw = -90.0f, pitch = 0.0f;
float sensitivity = 0.2f;
float radius = 10.0f;
bool firstMouse = true;
double lastX = 400, lastY = 300;
float fov = 45.0f;

glm::vec3 cameraPos;
glm::vec3 cameraUp(0.0f, 1.0f, 0.0f);

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

const char* fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;
uniform vec3 uColor;
void main() {
    FragColor = vec4(uColor, 1.0);
}
)";

GLuint shaderProgram;

// ==================== Callbacks ====================
void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (firstMouse) { lastX = xpos; lastY = ypos; firstMouse = false; }
    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;
    lastX = xpos; lastY = ypos;
    xoffset *= sensitivity; yoffset *= sensitivity;
    yaw += xoffset; pitch += yoffset;
    if (pitch > 89.0f) pitch = 89.0f;
    if (pitch < -89.0f) pitch = -89.0f;
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    radius -= yoffset;
    if (radius < 1.0f) radius = 1.0f;
    if (radius > 100.0f) radius = 100.0f;
}

// ==================== Funciones de utilidades ====================
GLuint compileShader(const char* vs, const char* fs) {
    auto compile = [](GLuint shader, const char* src) {
        glShaderSource(shader, 1, &src, NULL);
        glCompileShader(shader);
        GLint success;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            char log[512]; glGetShaderInfoLog(shader, 512, NULL, log);
            throw std::runtime_error(log);
        }
    };

    GLuint vertex = glCreateShader(GL_VERTEX_SHADER);
    compile(vertex, vs);
    GLuint fragment = glCreateShader(GL_FRAGMENT_SHADER);
    compile(fragment, fs);

    GLuint program = glCreateProgram();
    glAttachShader(program, vertex);
    glAttachShader(program, fragment);
    glLinkProgram(program);
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    return program;
}

Range calcularRangos(const std::string& filePath) {
    std::ifstream archivo(filePath);
    if (!archivo) throw std::runtime_error("No se pudo abrir: " + filePath);

    double x, y, z;
    double minX = 1e9, maxX = -1e9;
    double minY = 1e9, maxY = -1e9;
    double minZ = 1e9, maxZ = -1e9;

    std::string linea;
    while (std::getline(archivo, linea)) {
        std::stringstream ss(linea);
        if (ss >> x >> y >> z) {
            minX = std::min(minX, x); maxX = std::max(maxX, x);
            minY = std::min(minY, y); maxY = std::max(maxY, y);
            minZ = std::min(minZ, z); maxZ = std::max(maxZ, z);
        }
    }
    return {minX, maxX, minY, maxY, minZ, maxZ};
}

void cargarRangosOriginales(const std::vector<std::string>& archivosTxt, const std::string& carpetaTxt) {
    for (const auto& nombre : archivosTxt) {
        std::string ruta = carpetaTxt + "/" + nombre;
        Range r = calcularRangos(ruta);
        std::string baseName = nombre.substr(0, nombre.find_last_of(".")); // sin extensión
        baseName = baseName + ".txt"; // agregar .txt para que coincida con el nombre del archivo bin
        rangos[baseName] = r;
        std::cout << "Archivo: " << baseName << " Rangos: X(" << r.minX << "," << r.maxX
                  << ") Y(" << r.minY << "," << r.maxY << ") Z(" << r.minZ << "," << r.maxZ << ")\n";
    }
}

void loadModel(const std::string& fileName, ModelPart& part) {
    std::ifstream in(fileName, std::ios::binary);
    if (!in) throw std::runtime_error("No se pudo abrir: " + fileName);

    size_t nPoints, nTets;
    in.read(reinterpret_cast<char*>(&nPoints), sizeof(size_t));
    std::vector<Point> points(nPoints);
    in.read(reinterpret_cast<char*>(points.data()), nPoints * sizeof(Point));

    in.read(reinterpret_cast<char*>(&nTets), sizeof(size_t));
    std::vector<Tetrahedron> tets(nTets);
    in.read(reinterpret_cast<char*>(tets.data()), nTets * sizeof(Tetrahedron));

    // Desnormalizar
    std::string baseName = std::filesystem::path(fileName).stem().string();
    if (rangos.find(baseName) == rangos.end()) throw std::runtime_error("No hay rangos para " + baseName);
    Range r = rangos[baseName];

    auto desnormalizar = [&](Point& p) {
        p.x = ((p.x + 1) / 2.0) * (r.maxX - r.minX) + r.minX;
        p.y = ((p.y + 1) / 2.0) * (r.maxY - r.minY) + r.minY;
        p.z = ((p.z + 1) / 2.0) * (r.maxZ - r.minZ) + r.minZ;
    };

    for (auto& p : points) desnormalizar(p);
    for (auto& t : tets) {
        desnormalizar(t.p1);
        desnormalizar(t.p2);
        desnormalizar(t.p3);
        desnormalizar(t.p4);
    }

    // Buffers
    std::vector<float> pointData;
    for (auto& p : points) {
        pointData.push_back(p.x);
        pointData.push_back(p.y);
        pointData.push_back(p.z);
    }

    std::vector<float> lineData;
    for (auto& t : tets) {
        auto addEdge = [&](Point a, Point b) {
            lineData.insert(lineData.end(), { (float)a.x,(float)a.y,(float)a.z,
                                              (float)b.x,(float)b.y,(float)b.z });
        };
        addEdge(t.p1, t.p2); addEdge(t.p2, t.p3); addEdge(t.p3, t.p1);
        addEdge(t.p1, t.p4); addEdge(t.p2, t.p4); addEdge(t.p3, t.p4);
    }

    // Puntos
    glGenVertexArrays(1, &part.VAO_points);
    glGenBuffers(1, &part.VBO_points);
    glBindVertexArray(part.VAO_points);
    glBindBuffer(GL_ARRAY_BUFFER, part.VBO_points);
    glBufferData(GL_ARRAY_BUFFER, pointData.size() * sizeof(float), pointData.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Líneas
    glGenVertexArrays(1, &part.VAO_lines);
    glGenBuffers(1, &part.VBO_lines);
    glBindVertexArray(part.VAO_lines);
    glBindBuffer(GL_ARRAY_BUFFER, part.VBO_lines);
    glBufferData(GL_ARRAY_BUFFER, lineData.size() * sizeof(float), lineData.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    part.pointCount = points.size();
    part.lineCount = lineData.size() / 3;

    // Color aleatorio
    static std::mt19937 gen{ std::random_device{}() };
    std::uniform_real_distribution<float> dist(0.2f, 1.0f);
    part.color = glm::vec3(dist(gen), dist(gen), dist(gen));
}

void renderLoop(GLFWwindow* window) {
    while (!glfwWindowShouldClose(window)) {
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        cameraPos.x = radius * cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        cameraPos.y = radius * sin(glm::radians(pitch));
        cameraPos.z = radius * sin(glm::radians(yaw)) * cos(glm::radians(pitch));

        glm::mat4 model = glm::mat4(1.0f);
        glm::mat4 view = glm::lookAt(cameraPos, glm::vec3(0.0f), cameraUp);
        glm::mat4 proj = glm::perspective(glm::radians(fov), 800.0f / 600.0f, 0.1f, 200.0f);

        glUseProgram(shaderProgram);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(proj));

        for (auto& part : modelParts) {
            glUniform3fv(glGetUniformLocation(shaderProgram, "uColor"), 1, glm::value_ptr(part.color));

            glBindVertexArray(part.VAO_points);
            glPointSize(3.0f);
            glDrawArrays(GL_POINTS, 0, part.pointCount);

            glBindVertexArray(part.VAO_lines);
            glLineWidth(1.2f);
            glDrawArrays(GL_LINES, 0, part.lineCount);
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}

// ==================== MAIN ====================
int main() {
    try {
        if (!glfwInit()) return -1;
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        GLFWwindow* window = glfwCreateWindow(800, 600, "Visualizador Rana 3D", NULL, NULL);
        if (!window) { glfwTerminate(); return -1; }
        glfwMakeContextCurrent(window);

        glfwSetCursorPosCallback(window, mouse_callback);
        glfwSetScrollCallback(window, scroll_callback);
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

        gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
        glEnable(GL_DEPTH_TEST);

        shaderProgram = compileShader(vertexShaderSource, fragmentShaderSource);

        std::vector<std::string> nombrePuntoSeparado = {
            "puntos_tiff_bloodMasks.txt", "puntos_tiff_brainMasks.txt",
            "puntos_tiff_duodenumMasks.txt", "puntos_tiff_eyeMasks_grupo1.txt",
            "puntos_tiff_eyeMasks_grupo2.txt", "puntos_tiff_eyeRetnaMasks_grupo1.txt",
            "puntos_tiff_eyeRetnaMasks_grupo2.txt", "puntos_tiff_eyeWhiteMasks_grupo1.txt"
        };
            

        // Cargar rangos desde carpeta original de TXT
        cargarRangosOriginales(nombrePuntoSeparado, "puntos_separados"); // Cambia "carpeta_txt" por la ruta real

        // Leer todos los bin y desnormalizar
        std::string carpetaBin = "output/"; // Carpeta donde están los .bin
        for (const auto& nombre : nombrePuntoSeparado) {
            std::string baseName = nombre.substr(0, nombre.find_last_of(".")); // quitamos .txt
            //std::string baseName = "";
            std::string rutaBin = carpetaBin + baseName  + ".txt.bin";

            if (std::filesystem::exists(rutaBin)) {
                ModelPart part;
                loadModel(rutaBin, part);
                modelParts.push_back(part);
                std::cout << "Cargado: " << rutaBin << "\n";
            } else {
                std::cerr << "No encontrado: " << rutaBin << "\n";
            }
        }

        
        /*
        for (auto& entry : std::filesystem::directory_iterator("output")) {
            if (entry.path().extension() == ".bin") {
                ModelPart part;
                loadModel(entry.path().string(), part);
                modelParts.push_back(part);
                std::cout << "Cargado: " << entry.path() << "\n";
            }
        }
        */

        renderLoop(window);
        glfwTerminate();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
}
