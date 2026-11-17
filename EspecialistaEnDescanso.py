import argparse
import bz2
from collections import deque
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import wave

import cv2
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist

# Directorio base del script para que funcione sin importar desde qué carpeta se ejecute
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SHAPE_PREDICTOR_URL = "https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
PILOT_CONFIG_PATH = os.path.join(SCRIPT_DIR, "piloto_seleccionado.json")

# Intentar importar playsound; si falla, recurrir a reproductores nativos de Windows/macOS/Linux
try:
    from playsound import playsound
    AUDIO_AVAILABLE = True
except Exception:
    AUDIO_AVAILABLE = False

ALARM_LOCK = threading.Lock()
CURRENT_ALARM_PROC = [None]

# ==============================================================================
# ZONA DE CONFIGURACIÓN CENTRALIZADA DE ESTADOS, SENSIBILIDAD Y PERCLOS
# Ajustá desde acá todos los valores de los 3 estados (Activo, Somnoliento, Peligro)
# ==============================================================================
CONFIG_DETECTOR = {
    # --- 1. Ventana y umbrales de PERCLOS (%) y pesos de cada síntoma ---
    "perclos_ventana_seg": 30.0,            # Ventana deslizante en segundos para calcular PERCLOS
    "perclos_min_obs_seg": 5.0,             # Observación mínima tras calibrar antes de evaluar PERCLOS
    "perclos_umbral_somnoliento_pct": 15.0, # >= 15% activa el MODO SOMNOLIENTO (Naranja, sin alarma sonora)
    "perclos_umbral_peligro_pct": 25.0,     # >= 25% dispara PELIGRO + ALARMA (si cierra ojos o cabecea)
    "perclos_peso_ojo_cerrado": 1.00,       # Peso (100%) que aporta tener los ojos cerrados a PERCLOS
    "perclos_peso_ojo_entrecerrado": 0.25,  # Peso (25%) que aporta entrecerrar los ojos a PERCLOS
    "perclos_peso_cabeceo": 0.70,           # Peso (70%) que aporta la cabeza caída/inclinada a PERCLOS
    "perclos_bono_por_bostezo_pct": 10.0,   # % extra que suma cada bostezo confirmado al PERCLOS actual
    "perclos_tope_bostezos_pct": 20.0,      # Tope máximo de % por bostezos (nunca llega a 25% por sí solo)

    # --- 2. ESTADO 1: MODO ACTIVO (Conductor fresco -> Baja sensibilidad, tolerante) ---
    "activo_ojos_peligro_seg": 1.15,        # Segundos de ojos cerrados para sonar alarma estando fresco
    "activo_ojos_entrecerrados_seg": 0.65,  # Segundos de ojos entrecerrados para pasar a Modo Somnoliento
    "activo_cabeceo_peligro_seg": 1.75,     # Segundos de caída de cabeza para sonar alarma estando fresco
    "activo_cabeceo_sintoma_seg": 0.85,     # Segundos de cabeceo corto que activa Modo Somnoliento (sin alarma)
    "activo_pitch_factor": 1.45,            # Caída vertical requerida en Modo Activo (145% de Pitch_0)
    "activo_pitch_delta_min": 0.24,         # Incremento mínimo sobre Pitch_0 en Modo Activo
    "activo_roll_grados": 24.0,             # Inclinación lateral permitida en Modo Activo (±24°)

    # --- 3. ESTADO 2: MODO SOMNOLIENTO (Pre-alerta -> Alta sensibilidad a ojos y cabeceo) ---
    "somnoliento_memoria_seg": 45.0,        # Segundos que dura el Modo Somnoliento tras un síntoma (bostezo/cabeceo)
    "somnoliento_ojos_peligro_seg": 0.65,   # Segundos de ojos cerrados para sonar alarma estando somnoliento
    "somnoliento_cabeceo_peligro_seg": 0.90,# Segundos de caída de cabeza para sonar alarma estando somnoliento
    "somnoliento_pitch_factor": 1.28,       # Caída vertical requerida en Modo Somnoliento (128% de Pitch_0)
    "somnoliento_pitch_delta_min": 0.16,    # Incremento mínimo sobre Pitch_0 en Modo Somnoliento
    "somnoliento_roll_grados": 18.0,        # Inclinación lateral permitida en Modo Somnoliento (±18°)

    # --- 4. Parámetros de Ojos (EAR), Bostezo (MAR) y Recuperación ---
    "ear_factor_activo": 0.82,              # Proporción del EAR calibrado para considerar ojo bien abierto
    "ear_factor_cerrado": 0.68,             # Proporción del EAR calibrado para considerar ojo cerrado
    "bostezo_mar_umbral": 0.62,             # Apertura interna de boca (MAR) para detectar bostezo
    "bostezo_duracion_seg": 1.35,           # Segundos con boca abierta para confirmar bostezo (NO dispara alarma)
    "parpadeo_min_seg": 0.05,               # Duración mínima de cierre para contar 1 parpadeo
    "recuperacion_activo_seg": 0.25,        # Segundos con ojos abiertos para recuperar estado / cortar alarma
}
# ==============================================================================


def preprocess_gray_clahe(frame, clahe, use_clahe=True, face_box=None, target_lum=110.0):
    """
    Preprocesamiento en el dominio espacial (TDI) enfocado exclusivamente en el rostro:
      1. Convierte el frame BGR a escala de grises.
      2. Mide la iluminación ADENTRO del recuadro del rostro detectado (ignorando el fondo),
         ponderando la media (40%) y el percentil 75 (60%) para detectar reflejos fuertes
         o luz directa sobre un lado de la cara.
      3. Si hay exceso de luz (ref_lum > target_lum), comprime las altas luces quemadas
         y aplica Corrección Gamma (gamma > 1.0, hasta 3.0) para oscurecer y recuperar
         los bordes de párpados, pestañas y labios.
      4. Aplica CLAHE local para uniformizar sombras laterales.
    Retorna: (gray_procesada, gamma_aplicado, luminancia_rostro)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Si tenemos las coordenadas del rostro, medimos la luz SOLO en la cara (sin el fondo)
    if face_box is not None:
        fx1, fy1, fx2, fy2 = face_box
        fx1, fy1 = max(0, fx1), max(0, fy1)
        fx2, fy2 = min(w, fx2), min(h, fy2)
        roi = gray[fy1:fy2, fx1:fx2]
    else:
        roi = np.array([])

    if roi.size < 100:
        # Fallback al centro de la imagen solo cuando todavía no hay rostro en cuadro
        roi = gray[int(h * 0.25):int(h * 0.75), int(w * 0.30):int(w * 0.70)]

    mean_lum = float(np.mean(roi))
    p75_lum = float(np.percentile(roi, 75))
    # Ponderar el percentil 75 permite detectar cuando un foco de luz quema parte de la cara
    ref_lum = 0.40 * mean_lum + 0.60 * p75_lum

    if not use_clahe:
        return gray, 1.0, ref_lum

    # Calcular exponente gamma adaptativo para llevar ref_lum hacia target_lum (110)
    norm_ref = np.clip(ref_lum / 255.0, 0.05, 0.96)
    norm_target = np.clip(target_lum / 255.0, 0.05, 0.96)
    gamma = float(np.clip(np.log(norm_target) / np.log(norm_ref), 0.45, 3.00))

    # Construir LUT con compresión suave de altas luces (evita blancos saturados > 210) y Gamma
    x = np.arange(256, dtype=np.float32) / 255.0
    if gamma > 1.15:
        # Si hay mucha luz en la cara, atenuar el tope de blancos antes de aplicar la curva gamma
        x = np.where(x > 0.75, 0.75 + (x - 0.75) * 0.55, x)
    lut = np.clip((x ** gamma) * 255.0, 0, 255).astype(np.uint8)
    gray_gamma = cv2.LUT(gray, lut)

    # Aplicar CLAHE sobre la imagen ya compensada en exposición
    gray_enhanced = clahe.apply(gray_gamma)
    return gray_enhanced, gamma, ref_lum


def eye_aspect_ratio(eye):
    """
    Calcula el Eye Aspect Ratio (EAR) a partir de los 6 puntos característicos del ojo.
    eye: arreglo numpy de forma (6, 2)
    """
    # Distancias euclidianas verticales
    vertical_1 = dist.euclidean(eye[1], eye[5])
    vertical_2 = dist.euclidean(eye[2], eye[4])
    # Distancia euclidiana horizontal
    horizontal = dist.euclidean(eye[0], eye[3])

    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def mouth_aspect_ratio(inner_mouth):
    """
    Calcula el Mouth Aspect Ratio (MAR) usando los 8 puntos del labio interno (landmarks 60-67).
    inner_mouth: arreglo numpy de forma (8, 2)
      - Puntos verticales: (1, 7), (2, 6), (3, 5) -> landmarks (61,67), (62,66), (63,65)
      - Puntos horizontales (comisuras internas): (0, 4) -> landmarks (60, 64)
    """
    vertical_1 = dist.euclidean(inner_mouth[1], inner_mouth[7])
    vertical_2 = dist.euclidean(inner_mouth[2], inner_mouth[6])
    vertical_3 = dist.euclidean(inner_mouth[3], inner_mouth[5])
    horizontal = dist.euclidean(inner_mouth[0], inner_mouth[4])

    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2 + vertical_3) / (3.0 * horizontal)


def compute_head_pose_metrics(landmarks):
    """
    Estima la postura de la cabeza en 3 ejes (Pitch, Roll y Yaw) a partir de landmarks 2D:
      - roll_deg: ángulo de inclinación lateral hacia los hombros (en grados) con la línea interocular.
      - pitch_ratio: razón proyectiva vertical ||p30 - p27|| / ||p8 - p30|| (cabeceo hacia abajo).
      - yaw_ratio: asimetría horizontal (d_L - d_R) / (d_L + d_R) entre la punta de la nariz (p30)
        y los bordes laterales del rostro (p0 derecho, p16 izquierdo).
          * yaw_ratio ~ 0.0 -> mirando de frente
          * |yaw_ratio| > 0.18 -> rostro girado hacia un costado (un ojo queda más expuesto a cámara)
    """
    right_eye_center = landmarks[36:42].mean(axis=0)
    left_eye_center = landmarks[42:48].mean(axis=0)

    dy = left_eye_center[1] - right_eye_center[1]
    dx = left_eye_center[0] - right_eye_center[0]
    roll_deg = float(np.degrees(np.arctan2(dy, max(abs(dx), 1e-6))))

    nose_bridge = landmarks[27]  # Entrecejo
    nose_tip = landmarks[30]     # Punta de la nariz
    chin = landmarks[8]          # Mentón
    jaw_right = landmarks[0]     # Contorno lateral del lado del ojo derecho (36-41)
    jaw_left = landmarks[16]     # Contorno lateral del lado del ojo izquierdo (42-47)

    nose_len = dist.euclidean(nose_tip, nose_bridge)
    nose_to_chin = dist.euclidean(chin, nose_tip)
    pitch_ratio = nose_len / max(nose_to_chin, 1e-6)

    # Distancias horizontales desde la punta de la nariz a cada borde de la cara
    dist_r = max(float(nose_tip[0] - jaw_right[0]), 1.0)
    dist_l = max(float(jaw_left[0] - nose_tip[0]), 1.0)
    yaw_ratio = (dist_l - dist_r) / (dist_l + dist_r)

    return (
        roll_deg,
        pitch_ratio,
        yaw_ratio,
        tuple(nose_bridge.astype(int)),
        tuple(nose_tip.astype(int)),
        tuple(chin.astype(int)),
    )


def compute_perspective_compensated_ear(
    left_ear,
    right_ear,
    yaw_ratio,
    left_ear_base=None,
    right_ear_base=None,
    yaw_thresh=0.18,
):
    """
    Calcula el EAR compensado tanto para calibración en diagonal como ante giros laterales (Yaw):
      1. Normalización independiente por ojo (left_ear_base, right_ear_base):
         Si la cámara está ubicada en diagonal al conductor, el ojo más lejano tiene un EAR base
         menor que el ojo cercano. Al escalar cada ojo respecto a su propia apertura calibrada,
         ambos ojos quedan equiparados al 100% de apertura en la postura diagonal de manejo.
      2. Ponderación por ojo expuesto a la cámara según yaw_ratio:
         Cuando la cara está girada (|yaw_ratio| > yaw_thresh), pondera con 85% el ojo más
         expuesto a la cámara y 15% el ojo en escorzo. Así evita falsos positivos al girar
         con ojos abiertos y detecta con precisión si se duerme estando de costado.
    """
    if (
        left_ear_base is not None
        and right_ear_base is not None
        and left_ear_base > 1e-4
        and right_ear_base > 1e-4
    ):
        mean_base = 0.5 * (left_ear_base + right_ear_base)
        l_norm = left_ear * (mean_base / left_ear_base)
        r_norm = right_ear * (mean_base / right_ear_base)
    else:
        l_norm = left_ear
        r_norm = right_ear

    if yaw_ratio > yaw_thresh:
        # El lado de left_eye (42-47) está más expuesto hacia la cámara
        return 0.85 * l_norm + 0.15 * r_norm
    elif yaw_ratio < -yaw_thresh:
        # El lado de right_eye (36-41) está más expuesto hacia la cámara
        return 0.85 * r_norm + 0.15 * l_norm
    else:
        return 0.5 * (l_norm + r_norm)


def detect_sunglasses_tdi(frame_bgr, landmarks, yaw_ratio=0.0):
    """
    Detecta el uso de lentes de sol (gafas oscuras) mediante Tratamiento Digital de Imágenes
    (análisis radiométrico comparativo entre la región orbital del lente y la piel del pómulo):
      - Cuando el conductor cierra los ojos SIN lentes de sol, el párpado tiene tono de piel,
        por lo que el percentil 68 de la caja orbital mantiene una luminancia cercana a la
        piel de la mejilla (ratio ~ 0.75 a 1.00).
      - Cuando usa LENTES DE SOL, el cristal oscuro cubre toda la región orbital, haciendo caer
        drásticamente el percentil 68 respecto a la mediana de luminancia del pómulo iluminado
        justo debajo (ratio < 0.58 y diferencia > 26 niveles de gris).
    Retorna: (tiene_lentes_frame, ratio_orbita_pomulo)
    """
    raw_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = raw_gray.shape
    nose_tip_y = int(landmarks[30][1])

    def _side_ratio(eye_pts, brow_pts):
        ex_min = int(np.min(eye_pts[:, 0]))
        ex_max = int(np.max(eye_pts[:, 0]))
        ey_min = int(np.min(eye_pts[:, 1]))
        ey_max = int(np.max(eye_pts[:, 1]))
        by_min = int(np.min(brow_pts[:, 1]))

        eye_w = max(12, ex_max - ex_min)
        pad_x = int(0.15 * eye_w)

        # ROI orbital (donde se ubica el cristal del lente de sol, desde bajo ceja hasta infra-órbita)
        ox1 = max(0, ex_min - pad_x)
        ox2 = min(w, ex_max + pad_x)
        oy1 = max(0, int(0.35 * by_min + 0.65 * ey_min))
        oy2 = min(h, int(ey_max + 0.30 * eye_w))

        # ROI del pómulo/mejilla inmediatamente debajo del lente (referencia de iluminación de piel)
        cx1 = max(0, ex_min)
        cx2 = min(w, ex_max)
        cy1 = min(h, int(ey_max + 0.55 * eye_w))
        cy2 = min(h, max(cy1 + 8, min(nose_tip_y, int(ey_max + 1.25 * eye_w))))

        orb_roi = raw_gray[oy1:oy2, ox1:ox2]
        cheek_roi = raw_gray[cy1:cy2, cx1:cx2]
        if orb_roi.size < 25 or cheek_roi.size < 25:
            return 1.0, 0.0

        # Percentil 68 en órbita (ignora pestañas finas o pequeños brillos) vs mediana en pómulo
        orb_p68 = float(np.percentile(orb_roi, 68))
        cheek_p50 = float(np.median(cheek_roi))
        if cheek_p50 < 38.0:
            return 1.0, 0.0

        ratio = orb_p68 / max(cheek_p50, 1.0)
        diff = cheek_p50 - orb_p68
        return ratio, diff

    # Landmarks: ojo derecho (36-41) + ceja derecha (17-21) | ojo izquierdo (42-47) + ceja izquierda (22-26)
    r_ratio, r_diff = _side_ratio(landmarks[36:42], landmarks[17:22])
    l_ratio, l_diff = _side_ratio(landmarks[42:48], landmarks[22:27])

    if yaw_ratio > 0.18:
        ratio, diff = l_ratio, l_diff
    elif yaw_ratio < -0.18:
        ratio, diff = r_ratio, r_diff
    else:
        ratio = 0.5 * (l_ratio + r_ratio)
        diff = 0.5 * (l_diff + r_diff)

    is_dark_lens = (ratio < 0.58) and (diff > 26.0)
    return is_dark_lens, ratio


def load_pilot_anchor(config_path=PILOT_CONFIG_PATH):
    """
    Carga desde disco la zona/posición normalizada (nx, ny) seleccionada para el piloto,
    permitiendo recordar qué lugar corresponde al piloto entre ejecuciones o cambios de conductor.
    """
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            nx = float(data.get("anchor_nx", 0.5))
            ny = float(data.get("anchor_ny", 0.5))
            if 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0:
                return (nx, ny)
        except Exception:
            pass
    return None


def save_pilot_anchor(anchor_norm, config_path=PILOT_CONFIG_PATH):
    """
    Guarda en disco la posición normalizada (nx, ny) donde fue seleccionado el piloto.
    """
    if anchor_norm is None:
        return
    try:
        data = {
            "anchor_nx": round(float(anchor_norm[0]), 4),
            "anchor_ny": round(float(anchor_norm[1]), 4),
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def track_subject_face(gray, detector, last_face_box, persistence_counter, max_persistence=25):
    """
    Mantiene la referencia espacial del sujeto principal incluso cuando mira en diagonal
    o se duerme de costado.
    Retorna: (lista_faces, persistence_counter, modo_persistencia_activo)
    """
    pilot_faces, persistence_counter, is_persisted, _, _, _ = detect_and_track_pilot(
        gray,
        detector,
        last_face_box,
        persistence_counter,
        pilot_anchor_norm=None,
        max_persistence=max_persistence,
    )
    return pilot_faces, persistence_counter, is_persisted


def detect_and_track_pilot(
    gray,
    detector,
    last_face_box,
    persistence_counter,
    pilot_anchor_norm=None,
    max_persistence=25,
    frame_dt=None,
):
    """
    Detecta múltiples rostros en cámara, mantiene el seguimiento del PILOTO seleccionado
    (con persistencia lateral si gira la cabeza) y recuerda la zona seleccionada para el
    piloto independientemente de si cambia la persona en ese asiento o si hay acompañantes:
      1. Ordena todos los rostros detectados de izquierda a derecha (#1, #2, ...).
      2. Si ya venía siguiendo al piloto (last_face_box), busca el rostro que mantiene
         continuidad espacial dentro de un radio máximo (para NO saltar a un acompañante
         cuando el piloto gira la cabeza de costado o sale momentáneamente).
      3. Si el detector frontal pierde el rostro del piloto por giro lateral extremo,
         activa búsqueda local con upsampling=1 y persistencia por hasta max_persistence
         (en segundos reales si se pasa frame_dt, o en frames) sin perder los demás rostros.
      4. Si el asiento del piloto quedó libre y entra un nuevo piloto (o el mismo regresa)
         en la zona recordada (pilot_anchor_norm), lo engancha automáticamente como PILOTO.
    Retorna:
      (pilot_faces, persistence_counter, is_persisted, all_faces, pilot_idx, pilot_anchor_norm)
    """
    h, w = gray.shape
    raw_faces = list(detector(gray, 0))
    # Ordenar rostros de izquierda a derecha para numeración estable (#1, #2, ...)
    all_faces = sorted(raw_faces, key=lambda r: 0.5 * (r.left() + r.right()))
    step_inc = float(frame_dt) if frame_dt is not None else 1

    def _center(r):
        return 0.5 * (r.left() + r.right()), 0.5 * (r.top() + r.bottom())

    # 1. Si tenemos seguimiento activo del piloto en el frame previo
    if last_face_box is not None:
        lx1, ly1, lx2, ly2 = last_face_box
        prev_cx = 0.5 * (lx1 + lx2)
        prev_cy = 0.5 * (ly1 + ly2)
        bw = max(1, lx2 - lx1)
        bh = max(1, ly2 - ly1)
        max_jump_dist = max(135.0, 1.15 * max(bw, bh))

        if len(all_faces) > 0:
            best_idx = None
            best_dist = float("inf")
            for idx, r in enumerate(all_faces):
                cx, cy = _center(r)
                d = np.hypot(cx - prev_cx, cy - prev_cy)
                if d < best_dist:
                    best_dist = d
                    best_idx = idx

            if best_idx is not None and best_dist <= max_jump_dist:
                chosen = all_faces[best_idx]
                if pilot_anchor_norm is None:
                    cx, cy = _center(chosen)
                    pilot_anchor_norm = (cx / w, cy / h)
                return [chosen], 0, False, all_faces, best_idx, pilot_anchor_norm

        # Si el piloto no apareció en la detección directa (o solo se detectaron acompañantes lejos),
        # aplicar recuperación local / persistencia lateral sobre la posición del piloto
        if persistence_counter < max_persistence:
            pad_x = int(0.35 * bw)
            pad_y = int(0.35 * bh)
            rx1 = max(0, lx1 - pad_x)
            ry1 = max(0, ly1 - pad_y)
            rx2 = min(w, lx2 + pad_x)
            ry2 = min(h, ly2 + pad_y)

            roi = gray[ry1:ry2, rx1:rx2]
            if roi.size > 400:
                sub_faces = detector(roi, 1)
                if len(sub_faces) > 0:
                    best_sub = max(
                        sub_faces,
                        key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top()),
                    )
                    recovered_rect = dlib.rectangle(
                        rx1 + best_sub.left(),
                        ry1 + best_sub.top(),
                        rx1 + best_sub.right(),
                        ry1 + best_sub.bottom(),
                    )
                    return [recovered_rect], persistence_counter + step_inc, True, all_faces, None, pilot_anchor_norm

            persisted_rect = dlib.rectangle(int(lx1), int(ly1), int(lx2), int(ly2))
            return [persisted_rect], persistence_counter + step_inc, True, all_faces, None, pilot_anchor_norm

    # 2. Si no había seguimiento activo (inicio o cambio de piloto tras dejar el asiento libre)
    if len(all_faces) > 0:
        if pilot_anchor_norm is not None:
            ax = pilot_anchor_norm[0] * w
            ay = pilot_anchor_norm[1] * h
            best_idx = None
            best_dist = float("inf")
            for idx, r in enumerate(all_faces):
                cx, cy = _center(r)
                d = np.hypot(cx - ax, cy - ay)
                if d < best_dist:
                    best_dist = d
                    best_idx = idx

            # Si el rostro está dentro de la zona del asiento del piloto (o es el único rostro y está cerca),
            # seleccionarlo automáticamente aunque haya cambiado la persona del piloto
            max_anchor_radius = 0.34 * w
            if best_idx is not None and best_dist <= max_anchor_radius:
                chosen = all_faces[best_idx]
                return [chosen], 0, False, all_faces, best_idx, pilot_anchor_norm
            else:
                # Hay acompañantes en cámara pero el lugar seleccionado del piloto está vacío
                return [], persistence_counter, False, all_faces, None, pilot_anchor_norm
        else:
            # Primera vez sin selección previa: elegir el rostro de mayor área en primer plano
            best_idx = max(
                range(len(all_faces)),
                key=lambda i: (all_faces[i].right() - all_faces[i].left())
                * (all_faces[i].bottom() - all_faces[i].top()),
            )
            chosen = all_faces[best_idx]
            cx, cy = _center(chosen)
            pilot_anchor_norm = (cx / w, cy / h)
            return [chosen], 0, False, all_faces, best_idx, pilot_anchor_norm

    return [], persistence_counter, False, all_faces, None, pilot_anchor_norm


def ema_filter(current_val, prev_val, alpha, dt=None, ref_dt=0.04):
    """
    Filtro pasa-bajos temporal de Media Móvil Exponencial (EMA) independiente de los FPS:
      x_suave(t) = alpha_eff * x(t) + (1 - alpha_eff) * x_suave(t - 1)
    Si se pasa dt (frame_dt en segundos), ajusta alpha_eff = 1 - (1 - alpha)^(dt / ref_dt)
    para mantener exactamente la misma respuesta temporal sin importar los FPS de la PC.
    """
    if prev_val is None:
        return float(current_val)
    if dt is not None and dt > 0.0 and 0.0 < alpha < 1.0:
        alpha_eff = 1.0 - ((1.0 - alpha) ** (dt / ref_dt))
        alpha_eff = float(np.clip(alpha_eff, 0.05, 0.98))
    else:
        alpha_eff = float(alpha)
    return float(alpha_eff * current_val + (1.0 - alpha_eff) * prev_val)


def classify_eye_state(ear, ear_active_thresh=0.25, ear_sleep_thresh=0.21):
    """
    Clasifica la apertura ocular instantánea en 3 niveles (inspirado en driver_drowsiness.py):
      2 -> Activo / Ojos bien abiertos (EAR > 0.25)
      1 -> Fatiga inicial / Entrecerrados (0.21 < EAR <= 0.25)
      0 -> Ojos cerrados (EAR <= 0.21)
    """
    if ear > ear_active_thresh:
        return 2
    elif ear > ear_sleep_thresh:
        return 1
    else:
        return 0


def update_perclos(
    history,
    now_ts,
    is_closed,
    window_sec=30.0,
    min_obs_sec=5.0,
    max_frame_dt=0.5,
    yawn_timestamps=None,
    yawn_bonus_pct=0.0,
    yawn_max_bonus_pct=20.0,
):
    """
    Actualiza la ventana deslizante temporal y calcula PERCLOS (%) ponderado por duración real (dt):
      1. Cada muestra aporta su duración real dt = min(t_i - t_{i-1}, max_frame_dt) multiplicada por
         el peso instantáneo de fatiga (1.0 para ojo cerrado, o pesos configurables para cabeceo y
         ojo entrecerrado), eliminando el sesgo por variaciones de FPS.
      2. Exige un tiempo mínimo de observación efectiva (min_obs_sec, por defecto 5.0 s) tras la
         calibración para evitar que un único frame o parpadeo inicial dispare PERCLOS al 100%.
      3. Si se pasan bostezos recientes (yawn_timestamps), cada bostezo dentro de la ventana suma
         yawn_bonus_pct al porcentaje (con tope yawn_max_bonus_pct < 25% para activar el Modo
         Somnoliento sin disparar la alarma sonora por sí solo).
    """
    sample_weight = float(np.clip(float(is_closed), 0.0, 1.0))
    history.append((float(now_ts), sample_weight))
    while history and (now_ts - history[0][0]) > window_sec:
        history.popleft()

    yawn_bonus = 0.0
    if yawn_timestamps is not None and yawn_bonus_pct > 0.0:
        while yawn_timestamps and (now_ts - yawn_timestamps[0]) > window_sec:
            yawn_timestamps.popleft()
        yawn_bonus = min(len(yawn_timestamps) * float(yawn_bonus_pct), float(yawn_max_bonus_pct))

    if len(history) < 2:
        return 0.0

    closed_duration = 0.0
    total_duration = 0.0
    for i in range(1, len(history)):
        t_prev, _ = history[i - 1]
        t_curr, weight_curr = history[i]
        dt = t_curr - t_prev
        if dt <= 0.0:
            continue
        # Acotar dt máximo por si el rostro salió de cuadro y volvió segundos después
        dt = min(dt, max_frame_dt)
        total_duration += dt
        if weight_curr > 0.0:
            closed_duration += dt * weight_curr

    if total_duration < min_obs_sec:
        return float(min(100.0, yawn_bonus))

    base_perclos = (closed_duration / total_duration) * 100.0
    return float(min(100.0, base_perclos + yawn_bonus))


def ensure_alarm_wav(path="alarma.wav", sample_rate=44100):
    """
    Si no existe el archivo 'alarma.wav', sintetiza automáticamente un patrón
    de alarma sonora (3 pulsos bitonales de 880 Hz / 1320 Hz) en formato WAV PCM 16-bit.
    """
    if os.path.exists(path):
        return path

    pulses = []
    for i in range(3):
        freq = 880.0 if i % 2 == 0 else 1320.0
        duration = 0.18
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        # Envolvente suave para evitar chasquidos de audio
        envelope = np.ones_like(t)
        fade = int(0.015 * sample_rate)
        envelope[:fade] = np.linspace(0, 1, fade)
        envelope[-fade:] = np.linspace(1, 0, fade)
        tone = 0.65 * envelope * np.sin(2.0 * np.pi * freq * t)
        silence = np.zeros(int(sample_rate * 0.07))
        pulses.extend([tone, silence])

    audio = np.concatenate(pulses)
    pcm16 = np.int16(np.clip(audio, -1.0, 1.0) * 32767)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())

    print(f"[INFO] Archivo de sonido '{path}' generado automáticamente.")
    return path


def ensure_shape_predictor(path):
    """
    Verifica que exista el archivo 'shape_predictor_68_face_landmarks.dat'.
    Si no está presente en la computadora (por ejemplo al clonar de GitHub o pasar por ZIP),
    lo descarga automáticamente desde dlib.net (.bz2) y lo descomprime usando Python puro,
    funcionando en Windows, Linux y macOS sin herramientas externas.
    """
    if os.path.exists(path):
        return path

    bz2_path = path + ".bz2"
    print(f"[INFO] No se encontró '{os.path.basename(path)}'.")
    print(f"[INFO] Descargando modelo oficial de 68 landmarks (~61 MB) desde {SHAPE_PREDICTOR_URL} ...")

    def _report_progress(block_num, block_size, total_size):
        if total_size > 0:
            pct = min(block_num * block_size * 100.0 / total_size, 100.0)
            sys.stdout.write(f"\r[DESCAGANDO] {pct:5.1f}%")
            sys.stdout.flush()

    urllib.request.urlretrieve(SHAPE_PREDICTOR_URL, bz2_path, reporthook=_report_progress)
    print("\n[INFO] Descomprimiendo archivo .dat (esto demora unos segundos)...")
    with bz2.BZ2File(bz2_path, "rb") as src, open(path, "wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)

    try:
        os.remove(bz2_path)
    except OSError:
        pass

    print(f"[INFO] Modelo listo en: {path}")
    return path


def stop_alarm():
    """
    Detiene inmediatamente cualquier reproducción de alarma en curso apenas el conductor se despierta.
    """
    proc = CURRENT_ALARM_PROC[0]
    if proc is not None:
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception:
            pass
        CURRENT_ALARM_PROC[0] = None

    if platform.system() == "Windows":
        try:
            import winsound
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass


def sound_alarm(path):
    """
    Reproduce el archivo 'alarma.wav' en un hilo separado sin bloquear el video,
    compatible con Windows (winsound), macOS (afplay) y Linux (pw-play / paplay / aplay),
    permitiendo interrumpirlo al instante con stop_alarm() cuando el conductor se despierta.
    """
    if not ALARM_LOCK.acquire(blocking=False):
        return  # Ya hay una alarma sonando actualmente
    try:
        ensure_alarm_wav(path)
        sys_name = platform.system()
        cmd = None
        if sys_name == "Windows" and os.path.exists(path):
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            time.sleep(0.65)
            return
        elif sys_name == "Darwin" and shutil.which("afplay") and os.path.exists(path):
            cmd = ["afplay", path]
        elif shutil.which("pw-play") and os.path.exists(path):
            cmd = ["pw-play", path]
        elif shutil.which("paplay") and os.path.exists(path):
            cmd = ["paplay", path]
        elif shutil.which("aplay") and os.path.exists(path):
            cmd = ["aplay", "-q", path]

        if cmd is not None:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            CURRENT_ALARM_PROC[0] = proc
            proc.wait()
            CURRENT_ALARM_PROC[0] = None
        elif AUDIO_AVAILABLE and os.path.exists(path):
            playsound(path)
        else:
            for _ in range(3):
                print("\a", end="", flush=True)
                time.sleep(0.3)
    except Exception as e:
        print(f"[ERROR] Error reproduciendo audio: {e}")
    finally:
        CURRENT_ALARM_PROC[0] = None
        ALARM_LOCK.release()


def open_camera(preferred_index=0):
    """
    Intenta abrir el índice de cámara indicado; si falla en otra computadora
    (por cámaras virtuales o diferente orden de dispositivos), prueba índices 0, 1 y 2.
    """
    candidates = [preferred_index] + [i for i in (0, 1, 2) if i != preferred_index]
    for idx in candidates:
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(idx)
        else:
            cap = cv2.VideoCapture(idx)

        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                if idx != preferred_index:
                    print(f"[INFO] Cámara {preferred_index} no disponible; usando cámara {idx}.")
                return cap
            cap.release()
    raise RuntimeError(
        "No se pudo abrir ninguna cámara web (índices probados: 0, 1, 2). "
        "Verificá que la cámara esté conectada y no esté siendo usada por otro programa."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="EspecialistaEnDescanso: Sistema de detección de somnolencia en tiempo real"
    )
    parser.add_argument(
        "--shape",
        default=os.path.join(SCRIPT_DIR, "shape_predictor_68_face_landmarks.dat"),
        help="Ruta al modelo dlib shape_predictor_68_face_landmarks.dat",
    )
    parser.add_argument(
        "--alarm",
        default=os.path.join(SCRIPT_DIR, "alarma.wav"),
        help="Ruta al archivo de sonido de alarma (por defecto: alarma.wav)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Índice de la cámara / dispositivo de captura",
    )
    parser.add_argument(
        "--perclos-window",
        type=float,
        default=30.0,
        help="Ventana temporal en segundos para el cálculo de PERCLOS (por defecto: 30s)",
    )
    parser.add_argument(
        "--alarm-interval",
        type=float,
        default=2.0,
        help="Intervalo en segundos para repetir la alarma mientras persista el sueño/alerta (por defecto: 2.0s)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_alarm_wav(args.alarm)
    ensure_shape_predictor(args.shape)

    print("[INFO] Cargando detector facial y predictor de 68 landmarks...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.shape)

    # Inicializar operador CLAHE moderado para no saturar zonas brillantes (TDI)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    use_clahe = True

    # Índices de landmarks para ojos y boca (de CodigoMedium.py + inner_mouth para MAR)
    (l_start, l_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (r_start, r_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (m_start, m_end) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    (im_start, im_end) = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]

    # Umbrales analíticos inicializados desde la zona centralizada CONFIG_DETECTOR
    ear_factor_activo = CONFIG_DETECTOR["ear_factor_activo"]
    ear_factor_cerrado = CONFIG_DETECTOR["ear_factor_cerrado"]
    ear_active_thresh = 0.25
    ear_sleep_thresh = 0.21
    drowsy_thresh_sec = CONFIG_DETECTOR["activo_ojos_entrecerrados_seg"]
    sleep_thresh_sec = CONFIG_DETECTOR["activo_ojos_peligro_seg"]
    active_thresh_sec = CONFIG_DETECTOR["recuperacion_activo_seg"]

    # Umbral analítico de MAR (Mouth Aspect Ratio) y duración para bostezo (síntoma, NO dispara alarma)
    mar_yawn_thresh = CONFIG_DETECTOR["bostezo_mar_umbral"]
    yawn_thresh_sec = CONFIG_DETECTOR["bostezo_duracion_seg"]

    # Parámetros y ventana deslizante de PERCLOS (%) + aportes de cabeceo y bostezos
    perclos_window_sec = args.perclos_window if args.perclos_window != 30.0 else CONFIG_DETECTOR["perclos_ventana_seg"]
    perclos_min_obs_sec = CONFIG_DETECTOR["perclos_min_obs_seg"]
    perclos_drowsy_thresh = CONFIG_DETECTOR["perclos_umbral_somnoliento_pct"]
    perclos_sleep_thresh = CONFIG_DETECTOR["perclos_umbral_peligro_pct"]
    peso_ojo_cerrado = CONFIG_DETECTOR["perclos_peso_ojo_cerrado"]
    peso_ojo_entrecerrado = CONFIG_DETECTOR["perclos_peso_ojo_entrecerrado"]
    peso_cabeceo = CONFIG_DETECTOR["perclos_peso_cabeceo"]
    bono_bostezo_pct = CONFIG_DETECTOR["perclos_bono_por_bostezo_pct"]
    tope_bostezos_pct = CONFIG_DETECTOR["perclos_tope_bostezos_pct"]
    perclos_history = deque()
    yawn_timestamps = deque()
    perclos = 0.0

    # Estado del MODO SOMNOLIENTO (Estado 2: aumenta la sensibilidad a ojos cerrados y cabeceos)
    somnoliento_memoria_seg = CONFIG_DETECTOR["somnoliento_memoria_seg"]
    drowsy_mode_until = -1e9
    drowsy_mode_active = False

    # Umbrales analíticos de postura cefálica (dinámicos según esté en Modo Activo o Modo Somnoliento)
    pitch_nod_thresh = 1.00
    roll_tilt_thresh = CONFIG_DETECTOR["activo_roll_grados"]
    head_thresh_sec = CONFIG_DETECTOR["activo_cabeceo_peligro_seg"]
    head_symptom_sec = CONFIG_DETECTOR["activo_cabeceo_sintoma_seg"]

    sleep_duration = 0.0
    drowsy_duration = 0.0
    active_duration = 0.0

    # Contador de parpadeos (transición de ojo cerrado >= parpadeo_min_seg a ojo abierto)
    blink_min_sec = CONFIG_DETECTOR["parpadeo_min_seg"]
    blink_duration = 0.0
    total_blinks = 0

    yawn_duration = 0.0
    total_yawns = 0
    yawn_counted = False

    head_duration = 0.0
    total_nods = 0
    nod_counted = False

    # Estado de calibración automática inicial (3 segundos efectivos con rostro detectado, o tecla 'c')
    calibration_duration = 3.0
    is_calibrating = True
    calib_elapsed_sec = 0.0
    eye_calib_elapsed_sec = 0.0  # Para auto-calibrar los ojos si inició la calibración con lentes de sol
    calib_left_ears = []
    calib_right_ears = []
    calib_pitches = []
    calib_rolls = []
    calib_yaws = []

    left_ear_base = None
    right_ear_base = None
    ear_base = None
    pitch_base = None
    roll_base = 0.0
    yaw_base = 0.0

    # Estado de seguimiento espacial, selección de piloto multi-rostro y persistencia lateral en segundos
    last_face_box = None
    persistence_sec = 0.0
    max_persistence_sec = 1.0
    pilot_anchor_norm = load_pilot_anchor()
    all_detected_faces = []
    current_pilot_idx = None
    pending_click_pt = [None]  # Coordenadas (x, y) si el usuario hace click sobre un rostro
    mirror_camera = True       # Imagen de la cámara invertida horizontalmente (modo espejo)

    if pilot_anchor_norm is not None:
        print(
            f"[INFO] Selección de asiento del PILOTO recordada en zona "
            f"X={pilot_anchor_norm[0]*100:.0f}%, Y={pilot_anchor_norm[1]*100:.0f}% "
            f"(Click sobre un rostro o tecla 'p' para cambiar de piloto)."
        )

    # Filtro pasa-bajos temporal (EMA) para eliminar jitter (alternable con tecla 'f')
    use_ema = True
    alpha_ear = 0.60   # Suavizado ligero para EAR (mantiene baja latencia)
    alpha_slow = 0.35  # Suavizado más firme para movimientos lentos (boca y postura cefálica)
    ema_ear = None
    ema_mar = None
    ema_pitch = None
    ema_roll = None
    ema_yaw = None

    # Detección automática TDI de Lentes de Sol (y modo manual con tecla 'g': AUTO -> ON -> OFF)
    sunglasses_mode = "AUTO"  # "AUTO", "ON", "OFF"
    sunglasses_active = False
    sunglasses_timer = 0.0
    sunglasses_hysteresis_sec = 0.40  # Segundos (~0.4s) para activar/desactivar sin parpadeos

    status = "Iniciando..."
    status_color = (255, 255, 255)
    alarm_repeat_interval = max(0.5, float(args.alarm_interval))
    last_alarm_ts = -1e9

    def _maybe_trigger_alarm(now_mono):
        nonlocal last_alarm_ts
        if (now_mono - last_alarm_ts) >= alarm_repeat_interval and not ALARM_LOCK.locked():
            last_alarm_ts = now_mono
            threading.Thread(
                target=sound_alarm, args=(args.alarm,), daemon=True
            ).start()

    win_main = "Especialista en Descanso - Monitor"
    win_det = "Result of detector (68 Landmarks)"
    cv2.namedWindow(win_main)
    cv2.namedWindow(win_det)

    def _on_mouse_click(event, mx, my, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pending_click_pt[0] = (mx, my)

    cv2.setMouseCallback(win_main, _on_mouse_click)
    cv2.setMouseCallback(win_det, _on_mouse_click)

    print(
        "[INFO] Iniciando captura de video "
        "(Click/'p'/'1-9': elegir Piloto | 'c': recalibrar | 'e': CLAHE | 'f': EMA | 'm': espejo | 'q': salir)..."
    )
    cap = open_camera(args.camera)
    fps_time = time.monotonic()
    prev_frame_ts = fps_time

    def _select_face_as_pilot(rect, idx_label, frame_w=640, frame_h=480):
        nonlocal last_face_box, persistence_sec, pilot_anchor_norm
        rx1, ry1, rx2, ry2 = rect.left(), rect.top(), rect.right(), rect.bottom()
        last_face_box = (rx1, ry1, rx2, ry2)
        persistence_sec = 0.0
        cx = 0.5 * (rx1 + rx2)
        cy = 0.5 * (ry1 + ry2)
        pilot_anchor_norm = (
            float(np.clip(cx / frame_w, 0.0, 1.0)),
            float(np.clip(cy / frame_h, 0.0, 1.0)),
        )
        save_pilot_anchor(pilot_anchor_norm)
        print(
            f"[INFO] Rostro #{idx_label} seleccionado y guardado como PILOTO "
            f"(zona X={pilot_anchor_norm[0]*100:.0f}%, Y={pilot_anchor_norm[1]*100:.0f}%)."
        )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now_ts = time.monotonic()
        frame_dt = min(max(now_ts - prev_frame_ts, 0.0), 0.25)
        prev_frame_ts = now_ts

        # Redimensionar para mantener costo computacional uniforme (de CodigoMedium.py)
        frame = cv2.resize(frame, (640, 480))
        # Invertir horizontalmente la imagen de la cámara (modo espejo por defecto, tecla 'm' para alternar)
        if mirror_camera:
            frame = cv2.flip(frame, 1)

        # Preprocesamiento TDI midiendo la luz exclusivamente dentro del rostro del piloto
        gray, current_gamma, mean_lum = preprocess_gray_clahe(
            frame, clahe, use_clahe=use_clahe, face_box=last_face_box
        )

        # Usar la imagen preprocesada en la ventana del detector para visualizar el efecto de Gamma + CLAHE
        face_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Detección multi-rostro + seguimiento del PILOTO con memoria de selección y persistencia lateral (en segundos)
        (
            faces,
            persistence_sec,
            is_persisted,
            all_detected_faces,
            current_pilot_idx,
            pilot_anchor_norm,
        ) = detect_and_track_pilot(
            gray,
            detector,
            last_face_box,
            persistence_sec,
            pilot_anchor_norm=pilot_anchor_norm,
            max_persistence=max_persistence_sec,
            frame_dt=frame_dt,
        )

        # Procesar selección por Click izquierdo del mouse sobre cualquier rostro en cámara
        if pending_click_pt[0] is not None:
            mx, my = pending_click_pt[0]
            pending_click_pt[0] = None
            if len(all_detected_faces) > 0:
                chosen_i = None
                best_d = float("inf")
                for i, r in enumerate(all_detected_faces):
                    # Priorizar si el click cayó adentro del recuadro del rostro
                    if r.left() <= mx <= r.right() and r.top() <= my <= r.bottom():
                        chosen_i = i
                        best_d = 0.0
                        break
                    rcx = 0.5 * (r.left() + r.right())
                    rcy = 0.5 * (r.top() + r.bottom())
                    d = np.hypot(mx - rcx, my - rcy)
                    if d < best_d:
                        best_d = d
                        chosen_i = i
                if chosen_i is not None and best_d <= 120.0:
                    chosen_rect = all_detected_faces[chosen_i]
                    _select_face_as_pilot(chosen_rect, chosen_i + 1)
                    faces = [chosen_rect]
                    current_pilot_idx = chosen_i
                    is_persisted = False

        if not faces:
            last_face_box = None
            stop_alarm()
            sleep_duration = drowsy_duration = active_duration = 0.0
            yawn_duration = head_duration = blink_duration = 0.0
            yawn_counted = nod_counted = False
            no_face_msg = (
                f"CALIBRANDO ({int(min(calib_elapsed_sec / calibration_duration, 1.0) * 100)}%) - Esperando Piloto..."
                if is_calibrating
                else "SIN PILOTO EN CUADRO"
            )
            cv2.putText(
                frame,
                no_face_msg,
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 165, 255),
                2,
            )
            # Si hay una zona de piloto recordada y el asiento está temporalmente vacío, mostrar guía
            if pilot_anchor_norm is not None:
                ax = int(pilot_anchor_norm[0] * 640)
                ay = int(pilot_anchor_norm[1] * 480)
                gx1, gy1 = max(10, ax - 70), max(40, ay - 70)
                gx2, gy2 = min(630, ax + 70), min(450, ay + 70)
                cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (140, 140, 140), 1)
                cv2.putText(
                    frame,
                    "Zona PILOTO (vacia)",
                    (gx1, max(55, gy1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (180, 180, 180),
                    1,
                )

        # Dibujar los demás rostros detectados (acompañantes / copiloto) en ambas ventanas
        for idx, other_face in enumerate(all_detected_faces):
            if idx == current_pilot_idx:
                continue
            ox1, oy1, ox2, oy2 = (
                other_face.left(),
                other_face.top(),
                other_face.right(),
                other_face.bottom(),
            )
            other_color = (255, 190, 0)  # Celeste para distinguir acompañantes del piloto
            label_other = f"Rostro #{idx + 1} [Click/p]"
            cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), other_color, 1)
            cv2.putText(
                frame,
                label_other,
                (ox1, max(20, oy1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                other_color,
                1,
            )
            cv2.rectangle(face_frame, (ox1, oy1), (ox2, oy2), other_color, 1)
            cv2.putText(
                face_frame,
                label_other,
                (ox1, max(45, oy1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                other_color,
                1,
            )

        for face in faces:
            # Recuadro del PILOTO en ambas vistas (verde: directo | naranja: persistencia lateral)
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            box_color = (0, 165, 255) if is_persisted else (0, 255, 0)
            pilot_num = (current_pilot_idx + 1) if current_pilot_idx is not None else 1
            pilot_tag = (
                f"PILOTO #{pilot_num} (Persistencia {persistence_sec:.1f}s/{max_persistence_sec:.1f}s)"
                if is_persisted
                else f"PILOTO #{pilot_num}"
            )
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(
                frame,
                pilot_tag,
                (x1, max(22, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                box_color,
                2,
            )
            cv2.rectangle(face_frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(
                face_frame,
                pilot_tag,
                (x1, max(45, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                box_color,
                2,
            )

            # Extracción de los 68 landmarks
            shape = predictor(gray, face)
            landmarks = face_utils.shape_to_np(shape)

            # Actualizar last_face_box: si está en persistencia lateral, seguir suavemente el centroide de landmarks
            if is_persisted:
                bw, bh = x2 - x1, y2 - y1
                lm_cx, lm_cy = np.mean(landmarks[:, 0]), np.mean(landmarks[:, 1])
                nx1 = int(0.7 * x1 + 0.3 * (lm_cx - 0.5 * bw))
                ny1 = int(0.7 * y1 + 0.3 * (lm_cy - 0.5 * bh))
                last_face_box = (nx1, ny1, nx1 + bw, ny1 + bh)
            else:
                last_face_box = (x1, y1, x2, y2)

            # Extraer regiones oculares y calcular EAR individual de cada ojo
            left_eye = landmarks[l_start:l_end]
            right_eye = landmarks[r_start:r_end]
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)

            # Extraer boca y calcular MAR instantáneo (Mouth Aspect Ratio) con el labio interno
            mouth = landmarks[m_start:m_end]
            inner_mouth = landmarks[im_start:im_end]

            # Calcular postura cefálica instantánea en 3 ejes: Roll, Pitch y Yaw (giro lateral)
            (
                roll_raw,
                pitch_raw,
                yaw_raw,
                pt_bridge,
                pt_nose,
                pt_chin,
            ) = compute_head_pose_metrics(landmarks)

            # Compensar escorzo horizontal de la boca cuando gira la cabeza (Yaw) para no inflar MAR al hablar de costado
            mar_raw = mouth_aspect_ratio(inner_mouth) * max(0.75, 1.0 - 0.35 * abs(yaw_raw))

            # Calcular EAR instantáneo con calibración independiente por ojo y compensación por giro (Yaw)
            ear_raw = compute_perspective_compensated_ear(
                left_ear,
                right_ear,
                yaw_raw,
                left_ear_base=left_ear_base,
                right_ear_base=right_ear_base,
                yaw_thresh=0.18,
            )

            # Actualizar filtro pasa-bajos temporal (EMA) adaptado a frame_dt (independiente de FPS)
            ema_ear = ema_filter(ear_raw, ema_ear, alpha_ear, dt=frame_dt)
            ema_mar = ema_filter(mar_raw, ema_mar, alpha_slow, dt=frame_dt)
            ema_pitch = ema_filter(pitch_raw, ema_pitch, alpha_slow, dt=frame_dt)
            ema_roll = ema_filter(roll_raw, ema_roll, alpha_slow, dt=frame_dt)
            ema_yaw = ema_filter(yaw_raw, ema_yaw, alpha_slow, dt=frame_dt)

            if use_ema:
                ear = ema_ear
                mar = ema_mar
                pitch_ratio = ema_pitch
                roll_deg = ema_roll
                yaw_ratio = ema_yaw
            else:
                ear = ear_raw
                mar = mar_raw
                pitch_ratio = pitch_raw
                roll_deg = roll_raw
                yaw_ratio = yaw_raw

            # Detección de Lentes de Sol (TDI radiométrico Órbita vs. Pómulo + control con tecla 'g')
            raw_has_sunglasses, _sg_ratio = detect_sunglasses_tdi(frame, landmarks, yaw_ratio=yaw_ratio)
            if sunglasses_mode == "ON":
                sunglasses_active = True
            elif sunglasses_mode == "OFF":
                sunglasses_active = False
            else:
                if raw_has_sunglasses:
                    sunglasses_timer = min(sunglasses_hysteresis_sec, sunglasses_timer + frame_dt)
                    if sunglasses_timer >= sunglasses_hysteresis_sec:
                        sunglasses_active = True
                else:
                    sunglasses_timer = max(-sunglasses_hysteresis_sec, sunglasses_timer - frame_dt)
                    if sunglasses_timer <= -sunglasses_hysteresis_sec:
                        sunglasses_active = False

            # Fase de calibración personalizada (3 segundos efectivos con rostro detectado, o tecla 'c')
            if is_calibrating:
                calib_elapsed_sec += frame_dt
                if not sunglasses_active and left_ear > 0.14 and right_ear > 0.14:
                    calib_left_ears.append(left_ear)
                    calib_right_ears.append(right_ear)
                    eye_calib_elapsed_sec += frame_dt
                calib_pitches.append(pitch_raw)
                calib_rolls.append(roll_raw)
                calib_yaws.append(yaw_raw)

                progress = min(calib_elapsed_sec / calibration_duration, 1.0)
                status = f"CALIBRANDO... {int(progress * 100)}%"
                status_color = (255, 255, 0)  # Cian

                # Barra de progreso de calibración en la parte superior
                cv2.rectangle(frame, (10, 50), (310, 68), (200, 200, 200), 1)
                cv2.rectangle(
                    frame,
                    (11, 51),
                    (10 + int(progress * 299), 67),
                    (255, 255, 0),
                    -1,
                )
                cv2.putText(
                    frame,
                    "Mira en tu postura normal de manejo (3s)",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (255, 255, 0),
                    2,
                )

                if calib_elapsed_sec >= calibration_duration and len(calib_pitches) >= 5:
                    if len(calib_left_ears) >= 5:
                        left_ear_base = float(np.median(calib_left_ears))
                        right_ear_base = float(np.median(calib_right_ears))
                        ear_base = 0.5 * (left_ear_base + right_ear_base)
                        ear_active_thresh = round(ear_base * ear_factor_activo, 3)
                        ear_sleep_thresh = round(ear_base * ear_factor_cerrado, 3)
                    pitch_base = float(np.median(calib_pitches))
                    roll_base = float(np.median(calib_rolls))
                    yaw_base = float(np.median(calib_yaws))
                    pitch_nod_thresh = round(
                        max(
                            pitch_base * CONFIG_DETECTOR["activo_pitch_factor"],
                            pitch_base + CONFIG_DETECTOR["activo_pitch_delta_min"],
                            0.94,
                        ),
                        3,
                    )

                    is_calibrating = False
                    perclos_history.clear()
                    yawn_timestamps.clear()
                    perclos = 0.0
                    drowsy_mode_until = -1e9
                    drowsy_mode_active = False
                    sleep_duration = drowsy_duration = active_duration = 0.0
                    yawn_duration = head_duration = blink_duration = 0.0
                    yawn_counted = nod_counted = False
                    print(
                        f"[INFO] Calibración completada -> "
                        f"Cero Postural: Pitch0={pitch_base:.2f}, Roll0={roll_base:+.1f}°, Yaw0={yaw_base:+.2f} | "
                        f"Umb EAR: {ear_active_thresh:.3f}/{ear_sleep_thresh:.3f}"
                    )
            elif not sunglasses_active and left_ear_base is None:
                # Auto-calibración ocular diferida en segundo plano si calibró con lentes de sol y luego se los sacó
                if left_ear > 0.14 and right_ear > 0.14:
                    calib_left_ears.append(left_ear)
                    calib_right_ears.append(right_ear)
                    eye_calib_elapsed_sec += frame_dt
                    if eye_calib_elapsed_sec >= calibration_duration and len(calib_left_ears) >= 5:
                        left_ear_base = float(np.median(calib_left_ears))
                        right_ear_base = float(np.median(calib_right_ears))
                        ear_base = 0.5 * (left_ear_base + right_ear_base)
                        ear_active_thresh = round(ear_base * ear_factor_activo, 3)
                        ear_sleep_thresh = round(ear_base * ear_factor_cerrado, 3)
                        print(
                            f"[INFO] Auto-calibración ocular diferida completada (sin lentes) -> "
                            f"EAR_L0={left_ear_base:.3f}, EAR_R0={right_ear_base:.3f} | "
                            f"Umb EAR: {ear_active_thresh:.3f}/{ear_sleep_thresh:.3f}"
                        )

            # Determinar si está en MODO SOMNOLIENTO (Estado 2: mayor sensibilidad ante cabeceos y cierre ocular)
            drowsy_mode_active = (not is_calibrating) and (
                (now_ts < drowsy_mode_until) or (perclos >= perclos_drowsy_thresh)
            )
            if drowsy_mode_active:
                sleep_thresh_sec = CONFIG_DETECTOR["somnoliento_ojos_peligro_seg"]
                head_thresh_sec = CONFIG_DETECTOR["somnoliento_cabeceo_peligro_seg"]
                roll_tilt_thresh = CONFIG_DETECTOR["somnoliento_roll_grados"]
                if pitch_base is not None:
                    pitch_nod_thresh = round(
                        max(
                            pitch_base * CONFIG_DETECTOR["somnoliento_pitch_factor"],
                            pitch_base + CONFIG_DETECTOR["somnoliento_pitch_delta_min"],
                            0.88,
                        ),
                        3,
                    )
                else:
                    pitch_nod_thresh = 0.92
            else:
                sleep_thresh_sec = CONFIG_DETECTOR["activo_ojos_peligro_seg"]
                head_thresh_sec = CONFIG_DETECTOR["activo_cabeceo_peligro_seg"]
                roll_tilt_thresh = CONFIG_DETECTOR["activo_roll_grados"]
                if pitch_base is not None:
                    pitch_nod_thresh = round(
                        max(
                            pitch_base * CONFIG_DETECTOR["activo_pitch_factor"],
                            pitch_base + CONFIG_DETECTOR["activo_pitch_delta_min"],
                            0.94,
                        ),
                        3,
                    )
                else:
                    pitch_nod_thresh = 1.00

            # Desviaciones posturales respecto al "cero relativo" calibrado (soporta cámara en diagonal)
            delta_roll = roll_deg - roll_base
            delta_yaw = yaw_ratio - yaw_base
            head_anomaly = (pitch_ratio > pitch_nod_thresh) or (abs(delta_roll) > roll_tilt_thresh)

            # 1. Evaluación temporal de BOSTEZO (MAR): es SÍNTOMA de somnolencia (suma a PERCLOS y activa
            #    el Modo Somnoliento), pero NUNCA dispara la alarma sonora por sí solo.
            mouth_color = (255, 0, 0)  # Azul por defecto
            if not is_calibrating and mar > mar_yawn_thresh:
                yawn_duration += frame_dt
                mouth_color = (0, 165, 255)  # Naranja mientras abre la boca
                if yawn_duration >= yawn_thresh_sec:
                    mouth_color = (0, 140, 255)  # Naranja intenso al confirmar síntoma de bostezo
                    if not yawn_counted:
                        total_yawns += 1
                        yawn_counted = True
                        yawn_timestamps.append(now_ts)
                        drowsy_mode_until = max(
                            drowsy_mode_until, now_ts + somnoliento_memoria_seg
                        )
                    cv2.putText(
                        frame,
                        "SINTOMA: BOSTEZO (+PERCLOS / Modo Somnoliento)",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.62,
                        (0, 165, 255),
                        2,
                    )
            else:
                yawn_duration = 0.0
                yawn_counted = False

            # 2. Evaluación temporal de CABECEO (Pitch) e INCLINACIÓN LATERAL (Roll)
            axis_color = (255, 255, 0)  # Cian por defecto para el eje facial
            if not is_calibrating and head_anomaly:
                head_duration += frame_dt
                axis_color = (0, 165, 255)  # Naranja mientras empieza a inclinar/cabecear
                # Si cabecea un tiempo intermedio, activa el Modo Somnoliento aunque todavía no llegue al peligro
                if head_duration >= head_symptom_sec:
                    drowsy_mode_until = max(
                        drowsy_mode_until, now_ts + somnoliento_memoria_seg
                    )
                if head_duration >= head_thresh_sec:
                    axis_color = (0, 0, 255)  # Rojo al confirmar postura de peligro
                    if not nod_counted:
                        total_nods += 1
                        nod_counted = True
                    head_alert_msg = (
                        "PELIGRO: CABECEO DETECTADO!"
                        if pitch_ratio > pitch_nod_thresh
                        else "PELIGRO: INCLINACION LATERAL!"
                    )
                    cv2.putText(
                        frame,
                        head_alert_msg,
                        (10, 105),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.72,
                        (0, 0, 255),
                        2,
                    )
            else:
                head_duration = 0.0
                nod_counted = False

            # 3. Evaluación ocular (EAR) e integración unificada de PERCLOS (ojos + cabeceo + bostezos)
            if sunglasses_active:
                eye_state = 2
                eye_closed_raw = False
                eye_closed = False
                sleep_duration = 0.0
                drowsy_duration = 0.0
                blink_duration = 0.0
                eye_w = 0.0
            else:
                eye_state = classify_eye_state(ear, ear_active_thresh, ear_sleep_thresh)
                eye_closed_raw = (
                    classify_eye_state(ear_raw, ear_active_thresh, ear_sleep_thresh) == 0
                )
                eye_closed = eye_state == 0

                if not is_calibrating:
                    if eye_closed_raw:
                        blink_duration += frame_dt
                    else:
                        if blink_duration >= blink_min_sec:
                            total_blinks += 1
                        blink_duration = 0.0

                    if eye_closed:
                        sleep_duration += frame_dt
                        drowsy_duration = 0.0
                        active_duration = 0.0
                    elif eye_state == 1:
                        sleep_duration = 0.0
                        active_duration = 0.0
                        drowsy_duration += frame_dt
                        if drowsy_duration >= drowsy_thresh_sec:
                            drowsy_mode_until = max(
                                drowsy_mode_until, now_ts + somnoliento_memoria_seg
                            )
                    else:
                        drowsy_duration = 0.0
                        sleep_duration = 0.0
                        active_duration += frame_dt

                eye_w = (
                    peso_ojo_cerrado
                    if eye_closed
                    else (peso_ojo_entrecerrado if eye_state == 1 else 0.0)
                )

            if not is_calibrating:
                head_w = peso_cabeceo if head_anomaly else 0.0
                instant_fatigue_weight = min(1.0, max(eye_w, head_w))
                perclos = update_perclos(
                    perclos_history,
                    now_ts,
                    instant_fatigue_weight,
                    window_sec=perclos_window_sec,
                    min_obs_sec=perclos_min_obs_sec,
                    yawn_timestamps=yawn_timestamps,
                    yawn_bonus_pct=bono_bostezo_pct,
                    yawn_max_bonus_pct=tope_bostezos_pct,
                )
                if perclos >= perclos_drowsy_thresh:
                    drowsy_mode_until = max(
                        drowsy_mode_until, now_ts + 5.0
                    )
                drowsy_mode_active = (now_ts < drowsy_mode_until) or (
                    perclos >= perclos_drowsy_thresh
                )
            else:
                perclos = 0.0
                drowsy_mode_active = False

            # 4. Máquina de 3 Estados (1: ACTIVO | 2: MODO SOMNOLIENTO | 3: PELIGRO + ALARMA)
            eye_danger = (not is_calibrating) and (not sunglasses_active) and (
                sleep_duration >= sleep_thresh_sec
                or (perclos >= perclos_sleep_thresh and eye_state < 2)
            )
            head_danger = (not is_calibrating) and (
                head_duration >= head_thresh_sec
                or (perclos >= perclos_sleep_thresh and head_anomaly)
            )

            if not is_calibrating:
                if eye_danger or head_danger:
                    if sleep_duration >= sleep_thresh_sec:
                        status = "PELIGRO: DURMIENDO !!!"
                    elif head_duration >= head_thresh_sec:
                        status = "PELIGRO: CABECEO CRITICO!"
                    else:
                        status = "PELIGRO: SOMNOLENCIA (PERCLOS)!"
                    status_color = (0, 0, 255)  # Rojo en BGR
                    _maybe_trigger_alarm(now_ts)
                elif drowsy_mode_active or head_anomaly or mar > mar_yawn_thresh:
                    status = (
                        "MODO SOMNOLIENTO [Lentes] !"
                        if sunglasses_active
                        else "MODO SOMNOLIENTO (Fatiga) !"
                    )
                    status_color = (0, 165, 255)  # Naranja en BGR (sin alarma sonora)
                    stop_alarm()
                else:
                    if sunglasses_active or active_duration >= active_thresh_sec:
                        status = (
                            "Activo [Lentes de Sol]"
                            if sunglasses_active
                            else "Activo :)"
                        )
                        status_color = (0, 255, 0)  # Verde en BGR
                    stop_alarm()
            else:
                stop_alarm()

            # Visualización de contornos de ojos (violeta si usa lentes de sol, o verde según ojo expuesto)
            if sunglasses_active:
                left_eye_color = (220, 100, 180)
                right_eye_color = (220, 100, 180)
            elif yaw_ratio > 0.18:
                left_eye_color = (0, 255, 0)      # Ojo izquierdo dominante (frente a cámara)
                right_eye_color = (160, 160, 100) # Ojo derecho en escorzo (menor peso)
            elif yaw_ratio < -0.18:
                left_eye_color = (160, 160, 100)  # Ojo izquierdo en escorzo (menor peso)
                right_eye_color = (0, 255, 0)     # Ojo derecho dominante (frente a cámara)
            else:
                left_eye_color = (0, 255, 0)
                right_eye_color = (0, 255, 0)

            cv2.polylines(frame, [left_eye], True, left_eye_color, 1)
            cv2.polylines(frame, [right_eye], True, right_eye_color, 1)

            mouth_hull = cv2.convexHull(mouth)
            cv2.polylines(frame, [mouth_hull], True, mouth_color, 1)
            inner_mouth_hull = cv2.convexHull(inner_mouth)
            cv2.polylines(frame, [inner_mouth_hull], True, mouth_color, 1)

            # Eje vertical del rostro (entrecejo -> punta de nariz -> mentón)
            cv2.line(frame, pt_bridge, pt_nose, axis_color, 2)
            cv2.line(frame, pt_nose, pt_chin, axis_color, 2)
            cv2.circle(frame, pt_nose, 3, (0, 0, 255), -1)

            # Dibujar los 68 puntos en la ventana secundaria del detector (de driver_drowsiness.py)
            for (lx, ly) in landmarks:
                cv2.circle(face_frame, (lx, ly), 2, (0, 255, 255), -1)

            # Mostrar estado y métricas EAR / MAR / PERCLOS / Postura en pantalla
            cv2.putText(
                frame,
                status,
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                status_color,
                2,
            )
            if sunglasses_active:
                cv2.putText(
                    frame,
                    "EAR: LENTES (Pausado)",
                    (415, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (220, 120, 255),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    f"EAR: {ear:.2f}",
                    (450, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                )
            cv2.putText(
                frame,
                f"MAR: {mar:.2f}",
                (450, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Bostezos: {total_yawns}",
                (450, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 165, 255),
                2,
            )

            # Color dinámico e indicador visual para PERCLOS (integra ojos + cabeceos + bostezos)
            if perclos >= perclos_sleep_thresh:
                perclos_color = (0, 0, 255)
            elif perclos >= perclos_drowsy_thresh:
                perclos_color = (0, 165, 255)
            else:
                perclos_color = (0, 255, 0)
            perclos_text = f"PERCLOS: {perclos:.1f}%"

            cv2.putText(
                frame,
                perclos_text,
                (450, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                perclos_color,
                2,
            )
            # Barra de progreso de PERCLOS (escala 0% a 50% para visualización clara)
            bar_x, bar_y, bar_w, bar_h = 450, 135, 170, 12
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)
            fill_w = int(min(perclos / 50.0, 1.0) * bar_w)
            if fill_w > 0:
                cv2.rectangle(
                    frame,
                    (bar_x + 1, bar_y + 1),
                    (bar_x + fill_w, bar_y + bar_h - 1),
                    perclos_color,
                    -1,
                )

            # Métricas de Postura Cefálica (Pitch, dRoll relativo, Yaw con dYaw y contador de cabeceos)
            cv2.putText(
                frame,
                f"Pitch: {pitch_ratio:.2f}",
                (450, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                axis_color,
                2,
            )
            cv2.putText(
                frame,
                f"dRoll: {delta_roll:+.1f} deg",
                (450, 195),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                axis_color,
                2,
            )
            yaw_color = (0, 255, 255) if abs(yaw_ratio) > 0.18 else (200, 200, 200)
            cv2.putText(
                frame,
                f"Yaw:{yaw_ratio:+.2f} d:{delta_yaw:+.2f}",
                (435, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                yaw_color,
                2,
            )
            cv2.putText(
                frame,
                f"Cabeceos: {total_nods}",
                (450, 245),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 165, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Parpadeos: {total_blinks}",
                (450, 270),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Umb EAR: {ear_active_thresh:.2f}/{ear_sleep_thresh:.2f}",
                (430, 295),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )
            sens_text = (
                "Sensib: ALTA (Somnoliento)"
                if drowsy_mode_active
                else "Sensib: Normal (Activo)"
            )
            sens_color = (0, 165, 255) if drowsy_mode_active else (180, 230, 180)
            cv2.putText(
                frame,
                sens_text,
                (415, 318),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                sens_color,
                1,
            )

        # Indicador de cantidad de rostros detectados, selección de Piloto y estado de Lentes de Sol
        num_faces = len(all_detected_faces)
        sg_tag = f"Lentes:{'SI' if sunglasses_active else 'NO'}({sunglasses_mode}[g])"
        pilot_str = (
            f"Rostros:{num_faces} | Piloto:#{current_pilot_idx + 1} [Click/p] | {sg_tag}"
            if current_pilot_idx is not None
            else f"Rostros:{num_faces} | [Click/p] Piloto | {sg_tag}"
        )
        cv2.putText(
            frame,
            pilot_str,
            (10, 435),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 120, 255) if sunglasses_active else ((0, 255, 255) if num_faces > 1 else (200, 200, 200)),
            1,
        )

        # Cálculo y visualización de FPS e indicadores de controles ('c', 'e', 'f', 'p', 'm', 'g')
        now = time.monotonic()
        fps = 1.0 / max(now - fps_time, 1e-6)
        fps_time = now
        cv2.putText(
            frame,
            f"FPS: {fps:.1f} | [c]Calib | [m]Espejo",
            (10, 460),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

        ema_label = "EMA:ON[f]" if use_ema else "EMA:OFF[f]"
        ema_color = (0, 255, 255) if use_ema else (180, 180, 180)
        cv2.putText(
            frame,
            ema_label,
            (270, 460),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            ema_color,
            2,
        )

        clahe_label = "Gamma+CLAHE:ON[e]" if use_clahe else "Gamma+CLAHE:OFF[e]"
        clahe_color = (0, 255, 0) if use_clahe else (180, 180, 180)
        cv2.putText(
            frame,
            clahe_label,
            (395, 460),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            clahe_color,
            2,
        )
        cv2.putText(
            face_frame,
            f"{clahe_label} | Luz:{mean_lum:.0f} | Gamma:{current_gamma:.2f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            clahe_color,
            2,
        )

        # Mostrar ambas vistas: monitor principal y detector de landmarks sobre imagen preprocesada
        cv2.imshow(win_main, frame)
        cv2.imshow(win_det, face_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # 'q' o tecla ESC
            break
        elif key == ord("g"):
            modes = ["AUTO", "ON", "OFF"]
            sunglasses_mode = modes[(modes.index(sunglasses_mode) + 1) % len(modes)]
            print(f"[INFO] Modo Lentes de Sol: {sunglasses_mode}")
        elif key == ord("m"):
            mirror_camera = not mirror_camera
            print(f"[INFO] Modo espejo de cámara: {'ACTIVADO' if mirror_camera else 'DESACTIVADO'}")
        elif key == ord("p") or key == 9:  # 'p' o Tab: alternar cuál rostro es el Piloto
            if len(all_detected_faces) > 0:
                next_idx = (
                    0
                    if current_pilot_idx is None
                    else (current_pilot_idx + 1) % len(all_detected_faces)
                )
                _select_face_as_pilot(all_detected_faces[next_idx], next_idx + 1)
        elif ord("1") <= key <= ord("9"):
            sel_idx = key - ord("1")
            if sel_idx < len(all_detected_faces):
                _select_face_as_pilot(all_detected_faces[sel_idx], sel_idx + 1)
        elif key == ord("e"):
            use_clahe = not use_clahe
        elif key == ord("f"):
            use_ema = not use_ema
            state_str = "ACTIVADO (estabilidad)" if use_ema else "DESACTIVADO (máxima sensibilidad)"
            print(f"[INFO] Filtro EMA {state_str}")
        elif key == ord("c"):
            # Reiniciar calibración personalizada de 3 segundos (ambos ojos + cero postural 3D)
            stop_alarm()
            is_calibrating = True
            calib_elapsed_sec = 0.0
            eye_calib_elapsed_sec = 0.0
            left_ear_base = None
            right_ear_base = None
            calib_left_ears.clear()
            calib_right_ears.clear()
            calib_pitches.clear()
            calib_rolls.clear()
            calib_yaws.clear()
            perclos_history.clear()
            yawn_timestamps.clear()
            perclos = 0.0
            drowsy_mode_until = -1e9
            drowsy_mode_active = False
            sleep_duration = drowsy_duration = active_duration = 0.0
            yawn_duration = head_duration = blink_duration = 0.0
            yawn_counted = nod_counted = False
            print("[INFO] Recalibrando umbrales y cero postural 3D por 3 segundos...")

    stop_alarm()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

