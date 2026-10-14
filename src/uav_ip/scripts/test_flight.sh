#!/bin/bash
# ==============================================================================
# Script de prueba simplificado para validar el stack (XRCE + PX4 + Bridge + Pilot)
# ==============================================================================
set -e

echo "[1/4] Iniciando MicroXRCEAgent..."
MicroXRCEAgent udp4 -p 8888 > /tmp/xrce_test.log 2>&1 &
XRCE_PID=$!
sleep 2

echo "[2/4] Iniciando PX4 SITL y Gazebo..."
cd ~/PX4-Autopilot
export GZ_VERSION=harmonic
export PX4_GZ_WORLD=task_0000
export PX4_GZ_MODEL_POSE="0,0,0.2,0,0,0"
# Usamos HEADLESS=1 para evitar que la GUI consuma recursos si no la necesitas, 
# pero si quieres ver el simulador, puedes comentar la variable HEADLESS.
# export HEADLESS=1 

make px4_sitl gz_x500 > /tmp/px4_test.log 2>&1 &
PX4_PID=$!

echo "Esperando 40 segundos para que Gazebo y PX4 arranquen completamente..."
sleep 40

echo "[3/4] Verificando topicos ROS 2..."
echo "(Nota: Con el modelo gz_x500 y observacion privilegiada, no se requiere ros_gz_bridge para nube de camara)"
ros2 topic list

echo "[4/4] Lanzando el piloto experto..."
# Corremos el piloto en primer plano para ver sus logs en la terminal
ros2 run uav_ip waypoint_controller --ros-args -p waypoints_file:=/home/gabrielq/ros2_ws/src/uav_ip/generated_worlds/task_0000/waypoints.yaml

echo "Limpiando procesos..."
kill $PX4_PID $XRCE_PID || true
pkill -f 'gz sim' || true
pkill -f 'px4' || true
