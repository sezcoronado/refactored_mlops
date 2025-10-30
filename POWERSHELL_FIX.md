# 🔧 Script PowerShell Corregido - Solución de Problemas

## ❌ Problema Original

Si viste este error:

```
Run-All : The term 'Run-All' is not recognized as the name of a 
cmdlet, function, script file, or operable program.
```

**¡Ya está corregido!** 🎉

---

## ✅ Solución Aplicada

He corregido el script `docker-run.ps1` para que funcione correctamente en PowerShell.

### Cambios realizados:
- ❌ Nombres con guión: `Run-All`, `Run-EDA`, etc.
- ✅ Nombres sin guión: `RunAll`, `RunEDA`, etc.

**Razón:** PowerShell puede tener conflictos con nombres de funciones que usan guiones.

---

## 🚀 Cómo Usar (Actualizado)

### **Descarga el archivo actualizado:**

1. **[obesity-ml-project.zip](computer:///mnt/user-data/outputs/obesity-ml-project.zip)** (Windows)
2. **[obesity-ml-project.tar.gz](computer:///mnt/user-data/outputs/obesity-ml-project.tar.gz)** (Linux/Mac)

### **Descomprime y ejecuta:**

```powershell
# 1. Descomprimir
Expand-Archive obesity-ml-project.zip -DestinationPath .

# 2. Entrar al directorio
cd obesity-ml-project

# 3. Habilitar ejecución de scripts (solo primera vez)
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

# 4. Ejecutar workflow completo
.\docker-run.ps1 all
```

---

## ✅ Ahora Debería Funcionar

```powershell
PS> .\docker-run.ps1 all

# Output esperado:
🚀 Running complete workflow...

Step 1/3: Running EDA pipeline...
🔨 Building Docker images...
✅ Build complete!
📊 Running EDA pipeline...
✅ EDA pipeline complete!

Step 2/3: Comparing datasets...
🔍 Comparing datasets...
🎉 DATASETS ARE IDENTICAL! 🎉

Step 3/3: Running tests...
🧪 Running tests...
✅ 12 passed

╔═══════════════════════════════════════════════════════════╗
║        🎉 COMPLETE WORKFLOW FINISHED! 🎉                  ║
╚═══════════════════════════════════════════════════════════╝
```

---

## 📋 Todos los Comandos Disponibles

```powershell
.\docker-run.ps1 all       # Pipeline completo (EDA + Compare + Test)
.\docker-run.ps1 eda       # Solo EDA
.\docker-run.ps1 compare   # Solo comparación
.\docker-run.ps1 test      # Solo tests
.\docker-run.ps1 mlflow    # Abrir MLflow UI
.\docker-run.ps1 shell     # Shell interactivo
.\docker-run.ps1 build     # Construir imágenes
.\docker-run.ps1 clean     # Limpiar todo
.\docker-run.ps1 stop      # Detener contenedores
.\docker-run.ps1 logs      # Ver logs
.\docker-run.ps1 help      # Ver ayuda
```

---

## 🐛 Si Aún Tienes Problemas

### Problema 1: "Execution of scripts is disabled"

**Solución:**
```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Problema 2: "docker-compose not found"

**Solución:**
1. Asegúrate de que Docker Desktop esté instalado y corriendo
2. Verifica: `docker-compose --version`
3. Si no está instalado: https://docs.docker.com/compose/install/

### Problema 3: "Cannot connect to Docker daemon"

**Solución:**
1. Abre Docker Desktop
2. Espera a que esté completamente iniciado (icono verde)
3. Vuelve a intentar

### Problema 4: Script no hace nada

**Solución:**
```powershell
# Ver qué está pasando
.\docker-run.ps1 -Verbose all

# O ejecutar directamente:
docker-compose build
docker-compose run --rm eda-pipeline
```

---

## 🔄 Alternativa: Usar Docker Compose Directamente

Si el script sigue dando problemas, puedes usar Docker Compose directamente:

```powershell
# Construir
docker-compose build

# Ejecutar EDA
docker-compose run --rm eda-pipeline

# Comparar
docker-compose run --rm compare

# Tests
docker-compose run --rm test
```

---

## ✅ Verificación Final

Para verificar que todo está funcionando:

```powershell
# 1. Verificar Docker
docker --version
docker-compose --version

# 2. Verificar que Docker Desktop está corriendo
docker ps

# 3. Ejecutar el script
.\docker-run.ps1 all
```

---

## 📞 Soporte Adicional

Si sigues teniendo problemas:

1. **Verifica PowerShell version:**
   ```powershell
   $PSVersionTable.PSVersion
   # Debería ser 5.1 o superior
   ```

2. **Ejecuta con más información:**
   ```powershell
   .\docker-run.ps1 all -Verbose
   ```

3. **Revisa DOCKER_GUIDE.md** para más detalles

---

## 🎉 Resumen

- ✅ Script corregido y actualizado
- ✅ Descarga el nuevo archivo comprimido
- ✅ Ejecuta: `.\docker-run.ps1 all`
- ✅ Disfruta tu proyecto refactorizado

**¡Gracias por reportar el problema!** 😊
