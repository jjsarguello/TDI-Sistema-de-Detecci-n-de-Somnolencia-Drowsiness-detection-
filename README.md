# Sistema de Detección de Somnolencia en Tiempo Real (`EspecialistaEnDescanso.py`)

**UNRC – Tratamiento Digital de Imágenes (TDI)**  
**Grupo 2:** Pérez Rossi, Tomás | Argüello, Juan Salvador

Compatible con **Windows, Linux y macOS** (Python 3.9 a 3.13).

---

## ¿Qué archivos incluye el proyecto?
- **`EspecialistaEnDescanso.py`**: Sistema principal de detección de somnolencia y fatiga en tiempo real (métricas `EAR`, `MAR`, `PERCLOS` ponderado por duración real, postura cefálica 3D `Pitch`/`Roll`/`Yaw`, detección radiométrica de lentes de sol, seguimiento multi-rostro con selección de piloto, preprocesamiento `Gamma Adaptativo + CLAHE`, filtro temporal `EMA` independiente de los FPS y calibración biométrica automática).
- **`requirements.txt`**: Dependencias del proyecto.
- **`alarma.wav`**: Archivo de alerta sonora (si no se encuentra, el programa lo sintetiza automáticamente).
- **`shape_predictor_68_face_landmarks.dat`**: Modelo de 68 landmarks de `dlib` (pesa ~96 MB; **el programa lo descarga y descomprime automáticamente** la primera vez que se ejecuta si no está en la carpeta).
- **`Explicacion_EspecialistaEnDescanso.txt`**: Documentación detallada de cada módulo, fórmula y su utilidad en TDI.

---

## Descarga rápida del repositorio (`git clone`)

```bash
git clone https://github.com/jjsarguello/TDI-Sistema-de-Detecci-n-de-Somnolencia-Drowsiness-detection-.git
cd TDI-Sistema-de-Detecci-n-de-Somnolencia-Drowsiness-detection-
```

---

## Cómo ejecutar el programa en cualquier computadora

### Opción A: En Windows (PowerShell o CMD)
1. Abrir una terminal (**PowerShell** o **CMD**) dentro de la carpeta del proyecto.
2. Crear y activar un entorno virtual (recomendado):
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. Instalar las librerías (usa `dlib-bin` precompilado, no requiere instalar CMake ni C++):
   ```powershell
   pip install -r requirements.txt
   ```
4. Ejecutar el programa:
   ```powershell
   python EspecialistaEnDescanso.py
   ```

---

### Opción B: En Linux o macOS
1. Abrir una terminal dentro de la carpeta del proyecto.
2. Crear y activar un entorno virtual:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Instalar las librerías:
   ```bash
   pip install -r requirements.txt
   ```
4. Ejecutar el programa:
   ```bash
   python3 EspecialistaEnDescanso.py
   ```

---

## Controles interactivos dentro del programa
- **Calibración inicial automática:** Al iniciar, mirá normalmente a la cámara durante **3 segundos efectivos** para que el sistema calibre los umbrales según la apertura de cada ojo y tu postura 3D base (si iniciás con lentes de sol puestos y luego te los sacás, los ojos se auto-calibran automáticamente en segundo plano).
- **Click izquierdo sobre un rostro**: Seleccionar a esa persona como **PILOTO** y recordar su ubicación en el habitáculo.
- **Tecla `p` o `Tab`**: Alternar la selección de **PILOTO** entre todos los rostros detectados en cámara.
- **Teclas `1` a `9`**: Seleccionar directamente el rostro `#1`, `#2`, etc. como **PILOTO**.
- **Flechas del teclado (`↑`, `↓`, `←`, `→`)**: Mover el cabezal motorizado (**PTZ**) de la cámara de visión nocturna **Foscam FI9820W** para encuadrar al piloto.
- **Tecla `n`**: Alternar el modo de **Visión Nocturna Infrarroja (LEDs IR)** de la cámara FI9820W (`AUTO` $\rightarrow$ `ON` forzado $\rightarrow$ `OFF`).
- **Tecla `v`**: Alternar en vivo entre la **Cámara IP FI9820W** (Ethernet) y la **Webcam local** de la computadora.
- **Tecla `g`**: Cambiar modo de **Lentes de Sol** (`AUTO` por detección radiométrica TDI $\rightarrow$ `ON` forzado $\rightarrow$ `OFF`). Cuando está activo, omite `EAR` y evalúa somnolencia mediante postura 3D (`Pitch`/`dRoll`), bostezo (`MAR`) y `PERCLOS` postural/bostezos.
- **Tecla `m`**: Activar / Desactivar el modo espejo horizontal de la cámara.
- **Tecla `c`**: Volver a calibrar los umbrales (3 segundos efectivos con rostro en cuadro).
- **Tecla `e`**: Activar / Desactivar la corrección de iluminación en el rostro (**Gamma Adaptativo + CLAHE**).
- **Tecla `f`**: Activar / Desactivar el filtro de estabilidad temporal (**EMA** compensado por $\Delta t$).
- **Tecla `q` o `ESC`**: Salir del programa.

---

## Parámetros opcionales por línea de comandos
```bash
python EspecialistaEnDescanso.py --camera auto --perclos-window 30 --alarm-interval 2.0
```
- `--camera`: Fuente de video: `auto` (por defecto: intenta primero la Foscam FI9820W en Ethernet `10.42.0.203` y si no está conectada usa la webcam local `0`), `ip` (o una IP específica), o un índice `0`, `1`, `2`.
- `--perclos-window`: Ventana deslizante en segundos para el cálculo de `PERCLOS` (por defecto `30.0`).
- `--alarm-interval`: Intervalo en segundos para repetir la alarma sonora mientras persista el estado de somnolencia (por defecto `2.0`; se detiene inmediatamente al abrir los ojos o recuperar la postura).
- `--shape`: Ruta personalizada al modelo `shape_predictor_68_face_landmarks.dat`.
- `--alarm`: Ruta personalizada al archivo `.wav` de alarma.

