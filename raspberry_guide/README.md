# Clasificador de residuos — ejecución en Raspberry Pi 4

Clasificación de residuos en tiempo real con cámara, usando el modelo **YOLO26s-cls**
entrenado en el proyecto *clasificador-residuos-juliaca*.

| Dato del modelo | Valor |
|---|---|
| Arquitectura | YOLO26s-cls (clasificación) |
| Parámetros | 5.4 M |
| Cómputo | 12.1 GFLOPs |
| Peso del archivo | ~11 MB |
| Entrenado con | ultralytics 8.3.209 |

## Requisitos

- Raspberry Pi 4 (2 GB de RAM o más)
- **Raspberry Pi OS de 64 bits** (Bookworm o posterior). En 32 bits PyTorch no funciona.
  Para comprobarlo: `uname -m` debe mostrar `aarch64`.
- Una cámara: cualquier webcam USB o el módulo de cámara oficial de Raspberry Pi
- Conexión a internet (solo durante la instalación)

## 1. Copiar los archivos a la Raspberry

**Opción A — desde GitHub (recomendada):**
```bash
git clone https://github.com/USUARIO/clasificador-residuos-juliaca.git
cd clasificador-residuos-juliaca/raspberry_guide
```

**Opción B — con una memoria USB:** copiar la carpeta `raspberry_guide/` completa a la memoria,
conectarla a la Raspberry y copiarla a la carpeta personal (`/home/pi/`).

La carpeta debe quedar así:
```
raspberry_guide/
├── modelo/
│   └── etapa2_j1028_s43.pt
├── LEEME_PRIMERO.txt
├── INICIAR.sh            ← todo en uno
├── clasificar_camara.py
├── requirements.txt
└── README.md
```

## 2. Ejecutar (todo en uno)

Conectar la cámara y, desde una terminal abierta en la carpeta:
```bash
bash INICIAR.sh
```
- **Primera vez:** instala todo dentro de un entorno aislado (`.venv`) con la versión exacta de
  ultralytics que necesita el modelo (10–20 minutos) y después abre la cámara.
- **Siguientes veces:** abre la cámara directamente.

Se abre una ventana con la imagen de la cámara, la clase detectada y el porcentaje de confianza.

- `q` o `ESC`: salir
- `s`: guardar una captura

### Opciones

| Comando | Para qué sirve |
|---|---|
| `bash INICIAR.sh --rapido` | Convierte el modelo a NCNN la primera vez (más FPS en la Raspberry). El `.pt` original no se modifica. |
| `bash INICIAR.sh --camara 1` | Usar otra cámara si hay varias conectadas |
| `bash INICIAR.sh --sin-ventana` | Sin pantalla (por ejemplo, por SSH); el resultado sale en la terminal |
| `bash INICIAR.sh --umbral 0.7` | Exigir más confianza antes de mostrar una clase |

## Solución de problemas

| Problema | Solución |
|---|---|
| `No se encontró ninguna cámara` | Revisar la conexión y ejecutar `v4l2-ctl --list-devices`. Probar otro puerto USB. |
| `el sistema es de 32 bits` | Reinstalar la SD con *Raspberry Pi OS (64-bit)* usando Raspberry Pi Imager. |
| Va lento | Usar `--rapido`. Cerrar otros programas. Con un buen disipador, la Pi no reduce velocidad por temperatura. |
| Error al cargar el modelo | Borrar la carpeta `.venv` y volver a ejecutar `bash INICIAR.sh` para reinstalar la versión correcta. |
| `cannot connect to X server` | Se está ejecutando por SSH: usar `--sin-ventana`. |
