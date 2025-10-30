# 🐳 Guía Completa de Uso con Docker

## ¿Por qué usar Docker?

Docker te permite ejecutar el proyecto en un **entorno aislado y reproducible**, evitando:
- ❌ Conflictos de versiones de Python
- ❌ Problemas de dependencias
- ❌ Diferencias entre sistemas operativos
- ❌ "En mi máquina funciona" syndrome

Con Docker obtienes:
- ✅ Mismo entorno en todas las máquinas
- ✅ Instalación automática de dependencias
- ✅ Aislamiento completo
- ✅ Fácil de compartir y reproducir

## 📋 Prerequisitos

### Instalar Docker

#### Windows:
1. Descarga [Docker Desktop para Windows](https://www.docker.com/products/docker-desktop)
2. Ejecuta el instalador
3. Reinicia tu computadora
4. Abre Docker Desktop

#### Mac:
1. Descarga [Docker Desktop para Mac](https://www.docker.com/products/docker-desktop)
2. Arrastra Docker.app a Applications
3. Abre Docker desde Applications

#### Linux (Ubuntu/Debian):
```bash
# Instalar Docker
sudo apt-get update
sudo apt-get install docker.io docker-compose

# Agregar usuario al grupo docker
sudo usermod -aG docker $USER
newgrp docker

# Verificar instalación
docker --version
docker-compose --version
```

### Verificar instalación

```bash
# Verificar Docker
docker --version
# Output esperado: Docker version 20.x.x o superior

# Verificar Docker Compose
docker-compose --version
# Output esperado: Docker Compose version 2.x.x o superior
```

## 🚀 Inicio Rápido (3 Pasos)

### Opción 1: Usando Scripts Helper (RECOMENDADO)

#### Linux/Mac:

```bash
# 1. Dar permisos de ejecución
chmod +x docker-run.sh

# 2. Ejecutar workflow completo
./docker-run.sh all
```

#### Windows (PowerShell):

```powershell
# 1. Habilitar ejecución de scripts (una sola vez)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 2. Ejecutar workflow completo
.\docker-run.ps1 all
```

### Opción 2: Usando Docker Compose directamente

```bash
# 1. Construir imágenes
docker-compose build

# 2. Ejecutar pipeline EDA
docker-compose run --rm eda-pipeline

# 3. Comparar resultados
docker-compose run --rm compare

# 4. Ejecutar tests
docker-compose run --rm test
```

## 📚 Comandos Disponibles

### 🎯 Usando los Scripts Helper

#### Linux/Mac (`./docker-run.sh`):

```bash
# Pipeline completo (EDA + Compare + Test)
./docker-run.sh all

# Solo EDA
./docker-run.sh eda

# Solo comparación
./docker-run.sh compare

# Solo tests
./docker-run.sh test

# MLflow UI (mantener corriendo)
./docker-run.sh mlflow

# Shell interactivo
./docker-run.sh shell

# Ver logs
./docker-run.sh logs

# Detener contenedores
./docker-run.sh stop

# Limpiar todo
./docker-run.sh clean

# Ver ayuda
./docker-run.sh help
```

#### Windows (`.\docker-run.ps1`):

```powershell
# Los mismos comandos, pero con sintaxis PowerShell
.\docker-run.ps1 all
.\docker-run.ps1 eda
.\docker-run.ps1 compare
.\docker-run.ps1 test
.\docker-run.ps1 mlflow
.\docker-run.ps1 shell
.\docker-run.ps1 help
```

### 🔧 Usando Docker Compose directamente

```bash
# Construir imágenes
docker-compose build

# Servicios individuales
docker-compose run --rm eda-pipeline    # Run EDA
docker-compose run --rm compare         # Compare datasets
docker-compose run --rm test            # Run tests
docker-compose run --rm shell bash      # Interactive shell

# MLflow UI (en background)
docker-compose up -d mlflow
# Acceder en: http://localhost:5000

# Ver logs
docker-compose logs -f

# Detener servicios
docker-compose down

# Limpiar todo
docker-compose down --rmi all --volumes
```

## 🎬 Workflows Comunes

### 1. Primera Ejecución (Setup completo)

```bash
# Opción A: Con script helper
chmod +x docker-run.sh
./docker-run.sh build
./docker-run.sh all

# Opción B: Con docker-compose
docker-compose build
docker-compose run --rm eda-pipeline
docker-compose run --rm compare
docker-compose run --rm test
```

**Resultado esperado:**
```
✅ EDA pipeline complete!
🎉 DATASETS ARE IDENTICAL! 🎉
✅ 12 tests PASSED
```

### 2. Ejecutar solo el Pipeline

```bash
# Opción A
./docker-run.sh eda

# Opción B
docker-compose run --rm eda-pipeline
```

**Output:**
- Genera `data/interim/dataset_limpio_refactored.csv`
- Muestra logs del proceso de limpieza
- No hay valores faltantes

### 3. Validar Resultados

```bash
# Opción A
./docker-run.sh compare

# Opción B
docker-compose run --rm compare
```

**Output:**
```
🎉 DATASETS ARE IDENTICAL! 🎉
✓ Shape match: (2153, 17)
✓ Values match: 100%
```

### 4. Ejecutar Tests

```bash
# Opción A
./docker-run.sh test

# Opción B
docker-compose run --rm test
```

**Output:**
```
12 passed in X.XXs
```

### 5. Explorar con MLflow UI

```bash
# Opción A
./docker-run.sh mlflow
# Presiona Ctrl+C para detener

# Opción B
docker-compose up mlflow
```

**Acceder a:** http://localhost:5000

**Verás:**
- Experimentos registrados
- Métricas de cada run
- Parámetros utilizados
- Artefactos generados

### 6. Shell Interactivo (Desarrollo)

```bash
# Opción A
./docker-run.sh shell

# Opción B
docker-compose run --rm shell bash
```

**Dentro del shell puedes:**
```bash
# Ver estructura
ls -la

# Ejecutar scripts manualmente
python scripts/run_eda.py
python scripts/compare_datasets.py

# Explorar datos
python
>>> import pandas as pd
>>> df = pd.read_csv('data/interim/dataset_limpio_refactored.csv')
>>> df.head()

# Ejecutar tests específicos
pytest tests/test_comparison.py -v

# Salir
exit
```

## 🔍 Verificación de Resultados

### Verificar que todo funciona

```bash
# 1. Ejecutar workflow completo
./docker-run.sh all

# 2. Verificar archivos generados
ls -la data/interim/
# Deberías ver: dataset_limpio_refactored.csv

# 3. Verificar contenido del archivo
docker-compose run --rm shell bash -c "python3 -c \"
import pandas as pd
df = pd.read_csv('data/interim/dataset_limpio_refactored.csv')
print(f'Shape: {df.shape}')
print(f'Missing values: {df.isnull().sum().sum()}')
print(f'Columns: {list(df.columns)}')
\""
```

**Output esperado:**
```
Shape: (2153, 17)
Missing values: 0
Columns: ['Gender', 'Age', 'Height', 'Weight', ...]
```

## 🐛 Troubleshooting

### Problema: "Docker daemon no está corriendo"

**Solución:**
- Windows/Mac: Abre Docker Desktop
- Linux: `sudo systemctl start docker`

### Problema: "Permission denied" (Linux)

**Solución:**
```bash
sudo usermod -aG docker $USER
newgrp docker
# O ejecutar con sudo: sudo docker-compose ...
```

### Problema: "Port 5000 already in use"

**Solución:**
```bash
# Cambiar puerto en docker-compose.yml
# De: "5000:5000"
# A:  "5001:5000"

# O detener el servicio que usa el puerto 5000
lsof -i :5000  # Ver qué está usando el puerto
```

### Problema: Scripts no tienen permisos de ejecución

**Solución:**
```bash
chmod +x docker-run.sh
# O ejecutar: bash docker-run.sh all
```

### Problema: Errores de construcción de imagen

**Solución:**
```bash
# Limpiar caché y reconstruir
docker-compose down --rmi all
docker system prune -a
docker-compose build --no-cache
```

### Problema: Contenedores antiguos bloqueando

**Solución:**
```bash
# Detener y remover todo
docker-compose down
docker ps -a  # Ver contenedores
docker rm $(docker ps -a -q)  # Remover todos

# O usar el script
./docker-run.sh clean
./docker-run.sh build
```

## 📊 Estructura de Volúmenes

Docker monta estas carpetas locales en el contenedor:

```
./data       → /app/data       (datasets)
./models     → /app/models     (modelos entrenados)
./reports    → /app/reports    (reportes y figuras)
./mlruns     → /app/mlruns     (tracking MLflow)
```

**Los archivos generados dentro del contenedor aparecen en tu máquina local.**

## 🎯 Casos de Uso

### Caso 1: Desarrollo y Testing

```bash
# 1. Abrir shell
./docker-run.sh shell

# 2. Hacer cambios en el código
vim src/data/data_cleaner.py

# 3. Ejecutar tests
pytest tests/ -v

# 4. Si pasa, ejecutar pipeline completo
python scripts/run_eda.py
```

### Caso 2: Presentación/Demo

```bash
# 1. Iniciar MLflow UI
./docker-run.sh mlflow

# 2. En otra terminal, ejecutar pipeline
./docker-run.sh all

# 3. Mostrar resultados en MLflow
# Abrir navegador: http://localhost:5000
```

### Caso 3: Validación de Resultados

```bash
# Ejecutar y validar en un solo comando
./docker-run.sh all

# Output te dirá si todo está correcto
```

### Caso 4: CI/CD Pipeline

```bash
# En tu pipeline de CI/CD
docker-compose build
docker-compose run --rm test
if [ $? -eq 0 ]; then
    docker-compose run --rm eda-pipeline
    docker-compose run --rm compare
fi
```

## 🚀 Optimización y Tips

### Tip 1: Reducir tiempo de build

```bash
# Build solo una vez
docker-compose build

# Luego solo ejecutar
docker-compose run --rm eda-pipeline
```

### Tip 2: Ver logs en tiempo real

```bash
# En una terminal
docker-compose up mlflow

# En otra terminal
./docker-run.sh all
```

### Tip 3: Debugging

```bash
# Shell con código montado en vivo
docker-compose run --rm -v $(pwd)/src:/app/src shell bash

# Ahora puedes editar código en tu editor
# Y ejecutar inmediatamente en el contenedor
```

### Tip 4: Limpiar espacio en disco

```bash
# Limpiar contenedores e imágenes no usadas
docker system prune -a

# Limpiar volúmenes no usados
docker volume prune
```

## 📝 Resumen de Comandos

| Acción | Linux/Mac | Windows |
|--------|-----------|---------|
| Pipeline completo | `./docker-run.sh all` | `.\docker-run.ps1 all` |
| Solo EDA | `./docker-run.sh eda` | `.\docker-run.ps1 eda` |
| Comparar | `./docker-run.sh compare` | `.\docker-run.ps1 compare` |
| Tests | `./docker-run.sh test` | `.\docker-run.ps1 test` |
| MLflow UI | `./docker-run.sh mlflow` | `.\docker-run.ps1 mlflow` |
| Shell | `./docker-run.sh shell` | `.\docker-run.ps1 shell` |
| Limpiar | `./docker-run.sh clean` | `.\docker-run.ps1 clean` |

## ✅ Checklist de Verificación

Después de ejecutar con Docker, verifica:

- [ ] Archivo `data/interim/dataset_limpio_refactored.csv` existe
- [ ] Shape del dataset: (2153, 17)
- [ ] No hay valores faltantes: 0
- [ ] Comparación muestra: "DATASETS ARE IDENTICAL"
- [ ] Tests: 12/12 PASSED
- [ ] MLflow UI accesible en http://localhost:5000

## 🎉 Conclusión

Con Docker, el proyecto:
- ✅ Se ejecuta en cualquier sistema operativo
- ✅ Tiene dependencias aisladas
- ✅ Es 100% reproducible
- ✅ Está listo para producción

**Comando más simple:**
```bash
./docker-run.sh all
```

**Resultado esperado:**
```
🎉 COMPLETE WORKFLOW FINISHED! 🎉
✅ EDA pipeline complete!
✅ Datasets are identical!
✅ 12 tests passed!
```

---

**¿Necesitas ayuda?** Consulta la sección de Troubleshooting o abre un issue en el repositorio.
