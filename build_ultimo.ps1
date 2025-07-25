param(
    [string]$config = "Debug",
    [string]$opencv_path = "C:/opt/opencv-mingw-4.5.5",
    [string]$mingw_path = "C:/MinGW/bin",
    [string]$glfw_path = "C:/glfw-3.4.bin.WIN64",
    [string]$glad_path = "C:/glad",
    [string]$glm_path = "C:/opt/glm"
)

# Configurar variables de entorno
$env:CC = "$mingw_path/gcc.exe"
$env:CXX = "$mingw_path/g++.exe"
$env:PATH += ";$opencv_path/x64/mingw/bin;$glfw_path/lib-mingw-w64"

# Verificar y crear directorio build si no existe
if (!(Test-Path -Path "build")) {
    Write-Host "Creando directorio build..."
    New-Item -ItemType Directory -Path "build" | Out-Null
}

# Configurar proyecto con CMake
Write-Host "Configurando proyecto con CMake..."
cmake -DCMAKE_BUILD_TYPE=$config `
      -DOpenCV_DIR="$opencv_path/x64/mingw/lib" `
      -G "MinGW Makefiles" `
      -B build

# Compilar proyecto
Write-Host "Compilando proyecto..."
cmake --build build --config $config

# Ejecutar el programa
if ($LASTEXITCODE -eq 0) {
    Write-Host "`nEjecutando OpenGL_OpenCV_Integration.exe..."
    $exePath = Join-Path -Path (Get-Item -Path "build").FullName -ChildPath "OpenGL_OpenCV_Integration.exe"
    
    if (Test-Path -Path $exePath) {
        Start-Process -NoNewWindow -FilePath $exePath -Wait
    } else {
        Write-Host "Error: No se encontró el ejecutable $exePath" -ForegroundColor Red
    }
} else {
    Write-Host "Error en la compilación" -ForegroundColor Red
}

Write-Host "`nProceso completado."