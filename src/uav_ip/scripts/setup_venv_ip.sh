#!/usr/bin/env bash
# ==============================================================================
# setup_venv_ip.sh — Configura el entorno virtual aislado de ML (~/venv_ip)
#
# Desacopla las librerias de Machine Learning (PyTorch, PyTorch Geometric,
# rosbags, SciPy, NumPy, Matplotlib, PyYAML) respecto del entorno de sistema
# de ROS 2 Jazzy (Ubuntu 24.04 PEP 668), e instala el paquete `uav_ip` en modo
# editable sin dependencias de ROS.
#
# Uso:
#   bash src/uav_ip/scripts/setup_venv_ip.sh [--cpu]
# ==============================================================================
set -euo pipefail

VENV_PATH="${VENV_PATH:-$HOME/venv_ip}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
USE_CPU_TORCH=0

for arg in "$@"; do
    case "$arg" in
        --cpu)
            USE_CPU_TORCH=1
            ;;
    esac
done

echo "=================================================================="
echo " Configurando Entorno Virtual de ML para UAV Instant Policy"
echo "   Ruta del venv : $VENV_PATH"
echo "   Paquete uav_ip: $PKG_DIR"
echo "=================================================================="

# 1. Crear entorno virtual (con fallback si python3.12-venv / ensurepip no esta instalado)
if [ ! -x "$VENV_PATH/bin/python3" ]; then
    echo "[1/4] Creando entorno virtual en $VENV_PATH..."
    if python3 -m venv --system-site-packages "$VENV_PATH" 2>/dev/null; then
        echo "      Entorno creado con ensurepip."
    else
        echo "      ensurepip no disponible; creando venv con --without-pip --system-site-packages..."
        rm -rf "$VENV_PATH"
        python3 -m venv --without-pip --system-site-packages "$VENV_PATH"
    fi
else
    echo "[1/4] Entorno virtual existente detectado en $VENV_PATH."
fi

# Asegurar wrappers bin/pip y bin/pip3 dentro del venv
if [ ! -x "$VENV_PATH/bin/pip" ]; then
    cat << 'EOF' > "$VENV_PATH/bin/pip"
#!/usr/bin/env bash
exec "$(dirname "$0")/python3" -m pip "$@"
EOF
    chmod +x "$VENV_PATH/bin/pip"
    cp "$VENV_PATH/bin/pip" "$VENV_PATH/bin/pip3"
fi

# Crear directorio requerido por setup.cfg ([develop] script_dir=$base/lib/uav_ip)
mkdir -p "$VENV_PATH/lib/uav_ip"

# 2. Instalar dependencias cientificas y de Machine Learning
echo "[2/4] Instalando dependencias de ML (torch, torch_geometric, rosbags, scipy, numpy, matplotlib, pyyaml)..."
if [ "$USE_CPU_TORCH" -eq 1 ]; then
    "$VENV_PATH/bin/python3" -m pip install --upgrade \
        torch --index-url https://download.pytorch.org/whl/cpu
else
    "$VENV_PATH/bin/python3" -m pip install --upgrade torch
fi

"$VENV_PATH/bin/python3" -m pip install --upgrade \
    torch_geometric \
    rosbags \
    scipy \
    numpy \
    matplotlib \
    pyyaml

# 3. Instalar paquete uav_ip en modo editable sin aislamiento ni dependencias de ROS
echo "[3/4] Instalando paquete uav_ip en modo editable (--no-build-isolation --no-deps)..."
"$VENV_PATH/bin/python3" -m pip install --no-build-isolation --no-deps -e "$PKG_DIR"

# 4. Verificar criterio de aceptacion
echo "[4/4] Verificando importacion de modulos en $VENV_PATH..."
"$VENV_PATH/bin/python3" -c "import torch, torch_geometric, rosbags, scipy, numpy, matplotlib, yaml, uav_ip; print('OK')"

echo "=================================================================="
echo " Entorno ~/venv_ip configurado y validado exitosamente."
echo " Para activarlo manualmente: source $VENV_PATH/bin/activate"
echo "=================================================================="
