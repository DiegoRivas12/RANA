#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <limits>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <omp.h>
#include <opencv2/opencv.hpp>

// ------------------------- ESTRUCTURAS BÁSICAS -------------------------
struct Point {
    double x, y, z;
    bool operator<(const Point& other) const {
        return std::tie(x, y, z) < std::tie(other.x, other.y, other.z);
    }
    bool operator==(const Point& other) const {
        return std::abs(x - other.x) < 1e-6 && 
               std::abs(y - other.y) < 1e-6 && 
               std::abs(z - other.z) < 1e-6;
    }
};

struct Tetrahedron {
    Point p1, p2, p3, p4;
    
    bool operator==(const Tetrahedron& other) const {
        std::vector<Point> this_points = {p1, p2, p3, p4};
        std::vector<Point> other_points = {other.p1, other.p2, other.p3, other.p4};
        std::sort(this_points.begin(), this_points.end());
        std::sort(other_points.begin(), other_points.end());
        return this_points == other_points;
    }
};

// ------------------------- KD-TREE OPTIMIZADO -------------------------
class KDTree {
private:
    struct KDNode {
        Point point;
        int left = -1;
        int right = -1;
        int axis;
    };

    std::vector<KDNode> nodes;
    int rootIndex = -1;
    const double eps = 1e-6;

    int buildTree(std::vector<Point>& points, int depth = 0) {
        if (points.empty()) return -1;
        
        int axis = depth % 3;
        auto mid = points.begin() + points.size()/2;
        
        std::nth_element(points.begin(), mid, points.end(),
            [axis](const Point& a, const Point& b) {
                return axis == 0 ? a.x < b.x : (axis == 1 ? a.y < b.y : a.z < b.z);
            });
        
        int index = nodes.size();
        nodes.push_back({*mid, -1, -1, axis});
        
        std::vector<Point> leftPoints(points.begin(), mid);
        std::vector<Point> rightPoints(mid + 1, points.end());
        
        nodes[index].left = buildTree(leftPoints, depth + 1);
        nodes[index].right = buildTree(rightPoints, depth + 1);
        
        return index;
    }

    void nearestNeighbor(int nodeIdx, const Point& target, 
                        Point& best, double& bestDist, int depth = 0) const {
        if (nodeIdx == -1) return;

        const auto& node = nodes[nodeIdx];
        double dist = std::sqrt(
            (target.x - node.point.x) * (target.x - node.point.x) +
            (target.y - node.point.y) * (target.y - node.point.y) +
            (target.z - node.point.z) * (target.z - node.point.z)
        );

        if (dist < bestDist) {
            bestDist = dist;
            best = node.point;
        }

        double diff;
        if (node.axis == 0) diff = target.x - node.point.x;
        else if (node.axis == 1) diff = target.y - node.point.y;
        else diff = target.z - node.point.z;

        int near = diff <= 0 ? node.left : node.right;
        int far = diff <= 0 ? node.right : node.left;

        nearestNeighbor(near, target, best, bestDist, depth + 1);

        if (diff * diff < bestDist) {
            nearestNeighbor(far, target, best, bestDist, depth + 1);
        }
    }

public:
    void build(std::vector<Point> points) {
        nodes.clear();
        rootIndex = buildTree(points);
    }

    Point findNearest(const Point& target) const {
        if (rootIndex == -1) return target;
        Point best = nodes[rootIndex].point;
        double bestDist = std::numeric_limits<double>::max();
        nearestNeighbor(rootIndex, target, best, bestDist);
        return best;
    }
};

// ------------------------- DELAUNAY 3D OPTIMIZADO -------------------------
class Delaunay3D {
private:
    std::vector<Tetrahedron> tetrahedrons;
    std::vector<std::pair<Point, double>> circumsphereCache;
    std::vector<Point> points;
    KDTree pointTree;
    double R;
    const double eps = 1e-6;

    Tetrahedron create_super_tetrahedron() {
        return {
            {-R, -R, -R},
            {R, -R, -R},
            {0, R, -R},
            {0, 0, R}
        };
    }

    std::pair<Point, double> optimized_circumsphere(const Tetrahedron& t) {
        glm::vec3 a(t.p1.x, t.p1.y, t.p1.z);
        glm::vec3 b(t.p2.x, t.p2.y, t.p2.z);
        glm::vec3 c(t.p3.x, t.p3.y, t.p3.z);
        glm::vec3 d(t.p4.x, t.p4.y, t.p4.z);
        
        glm::vec3 ab = b - a;
        glm::vec3 ac = c - a;
        glm::vec3 ad = d - a;
        
        glm::vec3 cross_ac_ad = glm::cross(ac, ad);
        glm::vec3 cross_ad_ab = glm::cross(ad, ab);
        glm::vec3 cross_ab_ac = glm::cross(ab, ac);
        
        float denom = 2.0f * glm::dot(ab, cross_ac_ad);
        if (std::abs(denom) < 1e-6) {
            return {{0,0,0}, std::numeric_limits<double>::infinity()};
        }
        
        glm::vec3 center = a + (glm::dot(ab,ab) * cross_ac_ad
                            + (glm::dot(ac,ac) * cross_ad_ab)
                            + (glm::dot(ad,ad) * cross_ab_ac));
        center /= denom;
        
        double radius = glm::length(center - a);
        
        return {{center.x, center.y, center.z}, radius};
    }

    void update_circumsphere_cache() {
        circumsphereCache.resize(tetrahedrons.size());
        #pragma omp parallel for
        for (size_t i = 0; i < tetrahedrons.size(); ++i) {
            circumsphereCache[i] = optimized_circumsphere(tetrahedrons[i]);
        }
    }

public:
    Delaunay3D(double radius) : R(radius) {
        tetrahedrons.push_back(create_super_tetrahedron());
        points.push_back({-R, -R, -R});
        points.push_back({R, -R, -R});
        points.push_back({0, R, -R});
        points.push_back({0, 0, R});
        pointTree.build(points);
        update_circumsphere_cache();
    }

    void add_points_batch(const std::vector<Point>& newPoints) {
    // 1. Filtrado de puntos duplicados usando hash espacial
    std::unordered_map<size_t, Point> spatialHash;
    auto hashPoint = [](const Point& p) {
        return std::hash<double>()(std::floor(p.x/1e-6))*31 ^ 
               std::hash<double>()(std::floor(p.y/1e-6))*31 ^ 
               std::hash<double>()(std::floor(p.z/1e-6));
    };
    
    std::vector<Point> uniquePoints;
    for (const auto& p : newPoints) {
        size_t h = hashPoint(p);
        if (spatialHash.find(h) == spatialHash.end()) {
            spatialHash[h] = p;
            uniquePoints.push_back(p);
        }
    }
    
    // 2. Procesamiento paralelo de los puntos
    #pragma omp parallel
    {
        std::vector<std::pair<Point, std::vector<size_t>>> threadResults;
        
        #pragma omp for nowait
        for (size_t i = 0; i < uniquePoints.size(); ++i) {
            const auto& p = uniquePoints[i];
            std::vector<size_t> badTetrahedronIndices;
            
            // 3. Encontrar tetraedros inválidos usando la caché
            for (size_t j = 0; j < circumsphereCache.size(); ++j) {
                const auto& [center, radius] = circumsphereCache[j];
                double dist = std::sqrt(
                    (p.x-center.x)*(p.x-center.x) + 
                    (p.y-center.y)*(p.y-center.y) + 
                    (p.z-center.z)*(p.z-center.z));
                if (dist <= radius + eps) {
                    badTetrahedronIndices.push_back(j);
                }
            }
            
            if (!badTetrahedronIndices.empty()) {
                threadResults.emplace_back(p, badTetrahedronIndices);
            }
        }
        
        // 4. Procesar los resultados de cada hilo
        #pragma omp critical
{
    for (auto& [point, badIndices] : threadResults) {
        // 5. Añadir el punto a la lista global
        points.push_back(point);

        // 6. Crear un mapa para contar las caras
        std::unordered_map<std::string, std::array<Point, 3>> faceMap;
        std::unordered_map<std::string, int> faceCount;

        auto getFaceKey = [](const std::array<Point, 3>& face) {
            std::array<Point, 3> sortedFace = face;
            std::sort(sortedFace.begin(), sortedFace.end(), [](const Point& a, const Point& b) {
                return std::tie(a.x,a.y,a.z) < std::tie(b.x,b.y,b.z);
            });
            return std::to_string(sortedFace[0].x)+","+std::to_string(sortedFace[0].y)+","+std::to_string(sortedFace[0].z)+"|"+
                   std::to_string(sortedFace[1].x)+","+std::to_string(sortedFace[1].y)+","+std::to_string(sortedFace[1].z)+"|"+
                   std::to_string(sortedFace[2].x)+","+std::to_string(sortedFace[2].y)+","+std::to_string(sortedFace[2].z);
        };

        // Recorrer tetraedros malos y contar caras
        for (size_t idx : badIndices) {
            const auto& tet = tetrahedrons[idx];
            std::array<std::array<Point, 3>, 4> faces = {{
                {tet.p1, tet.p2, tet.p3},
                {tet.p1, tet.p2, tet.p4},
                {tet.p1, tet.p3, tet.p4},
                {tet.p2, tet.p3, tet.p4}
            }};
            for (auto& face : faces) {
                std::string key = getFaceKey(face);
                faceMap[key] = face;
                faceCount[key]++;
            }
        }

        // 7. Eliminar tetraedros malos
        tetrahedrons.erase(
            std::remove_if(tetrahedrons.begin(), tetrahedrons.end(),
                [&badIndices, this](const Tetrahedron& t) {
                    size_t index = &t - &tetrahedrons[0];
                    return std::find(badIndices.begin(), badIndices.end(), index) != badIndices.end();
                }),
            tetrahedrons.end()
        );

        // 8. Crear nuevos tetraedros desde las caras externas (aparecen una sola vez)
        for (const auto& [key, face] : faceMap) {
            if (faceCount[key] == 1) {
                // Validar que el tetraedro no tenga puntos repetidos
                if (!(face[0] == point || face[1] == point || face[2] == point)) {
                    Tetrahedron newTet = {face[0], face[1], face[2], point};
                    
                    // Verificar que los 4 puntos son distintos
                    if (!(newTet.p1 == newTet.p2 || newTet.p1 == newTet.p3 || newTet.p1 == newTet.p4 ||
                        newTet.p2 == newTet.p3 || newTet.p2 == newTet.p4 ||
                        newTet.p3 == newTet.p4)) {
                        tetrahedrons.push_back(newTet);
                    }
                }
            }
        }

    }
}

    }
    
    // 9. Reconstruir estructuras de datos auxiliares
    pointTree.build(points);
    update_circumsphere_cache();
}

    void remove_super_tetrahedron() {
        tetrahedrons.erase(
            std::remove_if(tetrahedrons.begin(), tetrahedrons.end(),
                [this](const Tetrahedron& t) {
                    Point super_vertices[4] = {{-R, -R, -R}, {R, -R, -R}, {0, R, -R}, {0, 0, R}};
                    int count = 0;
                    for (const auto& sv : super_vertices) {
                        if (t.p1 == sv || t.p2 == sv || t.p3 == sv || t.p4 == sv) {
                            count++;
                        }
                    }
                    return count > 0;
                }),
            tetrahedrons.end()
        );
        update_circumsphere_cache();
    }

    const std::vector<Tetrahedron>& get_tetrahedrons() const { return tetrahedrons; }
    const std::vector<Point>& get_points() const { return points; }
};

// ------------------------- SHADERS -------------------------
const char* vertexShaderSource = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 MVP;
void main() {
    gl_Position = MVP * vec4(aPos, 1.0);
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;
uniform vec3 color;
void main() {
    FragColor = vec4(color, 1.0);
}
)";

// ------------------------- VARIABLES GLOBALES -------------------------
GLuint shaderProgram;
GLuint VAO_points, VBO_points;
GLuint VAO_lines, VBO_lines, instanceVBO;
Delaunay3D delaunay3d(2);//0.675

// Control de cámara
float yaw = -90.0f, pitch = 0.0f;
float lastX = 400, lastY = 300;
float zoom = 10.0f;
bool firstMouse = true;

// ------------------------- FUNCIONES AUXILIARES -------------------------
GLuint compileShader(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, NULL);
    glCompileShader(shader);
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        glGetShaderInfoLog(shader, 512, NULL, log);
        std::cerr << "Error en shader: " << log << "\n";
    }
    return shader;
}

GLuint createShaderProgram() {
    GLuint vs = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);
    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    glDeleteShader(vs);
    glDeleteShader(fs);
    return program;
}

std::vector<Point> leerPuntosNormalizados(const std::string& ruta) {
    std::ifstream archivo(ruta);
    if (!archivo.is_open()) {
        std::cerr << "Error al abrir archivo\n";
        return {};
    }

    std::vector<Point> pts;
    double x, y, z;
    double minX = 1e9, maxX = -1e9, minY = 1e9, maxY = -1e9, minZ = 1e9, maxZ = -1e9;

    std::string linea;
    while (std::getline(archivo, linea)) {
        std::stringstream ss(linea);
        if (ss >> x >> y >> z) {
            pts.push_back({x, y, z});
            minX = std::min(minX, x); maxX = std::max(maxX, x);
            minY = std::min(minY, y); maxY = std::max(maxY, y);
            minZ = std::min(minZ, z); maxZ = std::max(maxZ, z);
        }
    }

    // Normalizar al rango [-1, 1]
    
    
    for (auto& p : pts) {
        p.x = ((p.x - minX) / (maxX - minX)) * 2 - 1;
        p.y = ((p.y - minY) / (maxY - minY)) * 2 - 1;
        p.z = ((p.z - minZ) / (maxZ - minZ)) * 2 - 1;
    }
    
    

    std::cout << "Puntos cargados: " << pts.size() << "\n";
    return pts;
}

void initOpenGL() {
    shaderProgram = createShaderProgram();
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

    // VAO/VBO para puntos
    glGenVertexArrays(1, &VAO_points);
    glGenBuffers(1, &VBO_points);
    glBindVertexArray(VAO_points);
    glBindBuffer(GL_ARRAY_BUFFER, VBO_points);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // VAO/VBO para líneas con instancing
    glGenVertexArrays(1, &VAO_lines);
    glGenBuffers(1, &VBO_lines);
    glGenBuffers(1, &instanceVBO);
    
    glBindVertexArray(VAO_lines);
    
    // Buffer para los vértices básicos de una línea (de -1 a 1 en X)
    float lineVertices[] = {-1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    glBindBuffer(GL_ARRAY_BUFFER, VBO_lines);
    glBufferData(GL_ARRAY_BUFFER, sizeof(lineVertices), lineVertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Buffer de instancias para las transformaciones de las líneas
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
    // Los atributos 1 y 2 serán para start y end de cada arista
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glVertexAttribDivisor(1, 1);
    
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glVertexAttribDivisor(2, 1);

    glEnable(GL_DEPTH_TEST);
    glPointSize(5.0f);
    glLineWidth(1.0f);
}
int numLineVertices = 0;

void updateBuffers() {
    // Datos de puntos
    std::vector<float> pointData;
    for (const auto& p : delaunay3d.get_points()) {
        pointData.push_back(static_cast<float>(p.x));
        pointData.push_back(static_cast<float>(p.y));
        pointData.push_back(static_cast<float>(p.z));
    }

    glBindBuffer(GL_ARRAY_BUFFER, VBO_points);
    glBufferData(GL_ARRAY_BUFFER, pointData.size() * sizeof(float), pointData.data(), GL_STATIC_DRAW);

    // Datos para las aristas (sin instancing)
    std::vector<float> edgeVertices;
    for (const auto& t : delaunay3d.get_tetrahedrons()) {
        const Point* pts[] = {&t.p1, &t.p2, &t.p3, &t.p4};
        int edges[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
        
        for (const auto& e : edges) {
            edgeVertices.push_back(static_cast<float>(pts[e[0]]->x));
            edgeVertices.push_back(static_cast<float>(pts[e[0]]->y));
            edgeVertices.push_back(static_cast<float>(pts[e[0]]->z));
            edgeVertices.push_back(static_cast<float>(pts[e[1]]->x));
            edgeVertices.push_back(static_cast<float>(pts[e[1]]->y));
            edgeVertices.push_back(static_cast<float>(pts[e[1]]->z));
        }
    }

    glBindVertexArray(VAO_lines);
    glBindBuffer(GL_ARRAY_BUFFER, VBO_lines);
    glBufferData(GL_ARRAY_BUFFER, edgeVertices.size() * sizeof(float), edgeVertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Guardar el total de vértices de las líneas
    numLineVertices = edgeVertices.size() / 3;
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (firstMouse) { lastX = xpos; lastY = ypos; firstMouse = false; }
    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;
    lastX = xpos;
    lastY = ypos;

    float sensitivity = 0.1f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    yaw += xoffset;
    pitch += yoffset;
    if (pitch > 89.0f) pitch = 89.0f;
    if (pitch < -89.0f) pitch = -89.0f;
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    zoom -= (float)yoffset * 0.5f;
    if (zoom < 1.0f) zoom = 1.0f;
    if (zoom > 50.0f) zoom = 50.0f;
}

void render(GLFWwindow* window) {
    std::cout << "Dibujando " << delaunay3d.get_points().size() << " puntos y " << numLineVertices/2 << " líneas.\n";

    while (!glfwWindowShouldClose(window)) {
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(shaderProgram);

        // Configuración de cámara
        glm::vec3 cameraPos = glm::vec3(
            zoom * cos(glm::radians(yaw)) * cos(glm::radians(pitch)),
            zoom * sin(glm::radians(pitch)),
            zoom * sin(glm::radians(yaw)) * cos(glm::radians(pitch))
        );
        glm::mat4 view = glm::lookAt(cameraPos, glm::vec3(0,0,0), glm::vec3(0,1,0));
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), 800.0f/600.0f, 0.1f, 100.0f);
        glm::mat4 MVP = projection * view;

        GLuint mvpLoc = glGetUniformLocation(shaderProgram, "MVP");
        glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, glm::value_ptr(MVP));

        // Dibujar puntos
        GLuint colorLoc = glGetUniformLocation(shaderProgram, "color");
        glUniform3f(colorLoc, 1.0f, 0.0f, 0.0f);
        glBindVertexArray(VAO_points);
        glDrawArrays(GL_POINTS, 0, delaunay3d.get_points().size());

        // Dibujar aristas con instancing
        // Dibujar aristas
        glUniform3f(colorLoc, 0.0f, 1.0f, 0.0f);
        glBindVertexArray(VAO_lines);
        glDrawArrays(GL_LINES, 0, numLineVertices);

        //std::cout << "Dibujando " << delaunay3d.get_tetrahedrons().size() * 6 << " aristas.\n";

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}

// ------------------------- MAIN -------------------------
int main() {
    // Cargar y procesar puntos
    //auto puntos = leerPuntosNormalizados("puntos_tiff_eye_mask_30000.txt");
    std::string pathPuntos = "puntos_generados";
    std::string pathSeparados = "puntos_separados";
    std::vector<std::string> nombrePunto = {"puntos_tiff_bloodMasks.txt", "puntos_tiff_brainMasks.txt",
    "puntos_tiff_duodenumMasks.txt", "puntos_tiff_eyeMasks.txt", "puntos_tiff_eyeRetnaMasks.txt",
    "puntos_tiff_eyeWhiteMasks.txt", "puntos_tiff_heartMasks.txt", "puntos_tiff_ileumMasks.txt",
    "puntos_tiff_kidneyMasks.txt", "puntos_tiff_lIntestineMasks.txt", "puntos_tiff_liverMasks.txt",
    "puntos_tiff_lungMasks.txt", "puntos_tiff_muscleMasks.txt", "puntos_tiff_nerveMasks.txt",
    "puntos_tiff_skeletonMasks.txt", "puntos_tiff_spleenMasks.txt",
    "puntos_tiff_stomachMasks.txt"};
    
    std::vector<std::string> nombrePuntoSeparado = {
    "puntos_separados/puntos_tiff_bloodMasks.txt",
    "puntos_separados/puntos_tiff_brainMasks.txt",
    "puntos_separados/puntos_tiff_duodenumMasks.txt",
    "puntos_separados/puntos_tiff_eyeMasks_grupo1.txt",
    "puntos_separados/puntos_tiff_eyeMasks_grupo2.txt",
    "puntos_separados/puntos_tiff_eyeRetnaMasks_grupo1.txt",
    "puntos_separados/puntos_tiff_eyeRetnaMasks_grupo2.txt",
    "puntos_separados/puntos_tiff_eyeWhiteMasks_grupo1.txt",
    "puntos_separados/puntos_tiff_eyeWhiteMasks_grupo2.txt",
    "puntos_separados/puntos_tiff_heartMasks.txt",
    "puntos_separados/puntos_tiff_ileumMasks.txt",
    "puntos_separados/puntos_tiff_kidneyMasks_grupo1.txt",
    "puntos_separados/puntos_tiff_kidneyMasks_grupo2.txt",
    "puntos_separados/puntos_tiff_lIntestineMasks.txt",
    "puntos_separados/puntos_tiff_liverMasks.txt",
    "puntos_separados/puntos_tiff_lungMasks_grupo1.txt",
    "puntos_separados/puntos_tiff_lungMasks_grupo2.txt",
    "puntos_separados/puntos_tiff_muscleMasks.txt",
    "puntos_separados/puntos_tiff_nerveMasks.txt",
    "puntos_separados/puntos_tiff_skeletonMasks.txt",
    "puntos_separados/puntos_tiff_spleenMasks.txt",
    "puntos_separados/puntos_tiff_stomachMasks.txt"};

    
    for(int i = 0; i < nombrePuntoSeparado.size(); i++) {
        std::string rutaCompleta = pathSeparados + "/" + nombrePunto[i];
        auto puntos = leerPuntosNormalizados(rutaCompleta);
        delaunay3d.add_points_batch(puntos);
    }
    // Procesamiento por lotes optimizado
     std::vector<Point> normales ={
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
        {-1.0, -1.0, -1.0},
        {1.0, 1.0, 1.0},
        {0.5, 0.5, 0.5},
        {-0.5, -0.5, -0.5},
        
        
    };
    std::vector<Point> puntos_coplanar = {
        {0.0, 5.0, 0.0},
        {4.33, 2.5, 0.0},
        {4.33, -2.5, 0.0},
        {0.0, -5.0, 0.0},
        {-4.33, -2.5, 0.0},
        {-4.33, 2.5, 0.0},
        {0.0, 0.0, 0.0}
    };
   
    //delaunay3d.add_points_batch(puntos);
    
    std::cout << "Total tetraedros: " << delaunay3d.get_tetrahedrons().size() << std::endl;
    /*
    for (const auto& t : delaunay3d.get_tetrahedrons()) {
        std::cout << "(" << t.p1.x << "," << t.p1.y << "," << t.p1.z << ") - "
                << "(" << t.p2.x << "," << t.p2.y << "," << t.p2.z << ") - "
                << "(" << t.p3.x << "," << t.p3.y << "," << t.p3.z << ") - "
                << "(" << t.p4.x << "," << t.p4.y << "," << t.p4.z << ")\n";
    }
    */
    

    //delaunay3d.add_points_batch(puntos);
    delaunay3d.remove_super_tetrahedron();
    
    // Inicializar GLFW y OpenGL
    if (!glfwInit()) return -1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    GLFWwindow* window = glfwCreateWindow(800, 600, "Delaunay 3D Ultra Optimizado", NULL, NULL);
    if (!window) { glfwTerminate(); return -1; }
    
    glfwMakeContextCurrent(window);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    
    // Configurar callbacks
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // Inicializar y renderizar
    initOpenGL();
    updateBuffers();
    render(window);

    // Limpieza
    glDeleteVertexArrays(1, &VAO_points);
    glDeleteBuffers(1, &VBO_points);
    glDeleteVertexArrays(1, &VAO_lines);
    glDeleteBuffers(1, &VBO_lines);
    glDeleteBuffers(1, &instanceVBO);
    glDeleteProgram(shaderProgram);
    glfwTerminate();
    return 0;
}