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

make px4_sitl gz_x500_depth > /tmp/px4_test.log 2>&1 &
PX4_PID=$!

echo "Esperando 40 segundos para que Gazebo y PX4 arranquen completamente..."
sleep 40

echo "[3/4] Iniciando ros_gz_bridge..."
ros2 run ros_gz_bridge parameter_bridge "/depth_camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked" > /tmp/bridge_test.log 2>&1 &
BRIDGE_PID=$!
sleep 5

echo "Topicos ROS 2 disponibles actualmente:"
ros2 topic list

echo "[4/4] Lanzando el piloto experto..."
# Corremos el piloto en primer plano para ver sus logs en la terminal
ros2 run uav_ip expert_pilot --ros-args -p waypoints_file:=/home/gabrielq/ros2_ws/src/uav_ip/generated_worlds/task_0000/waypoints.yaml

echo "Limpiando procesos..."
kill $BRIDGE_PID $PX4_PID $XRCE_PID
pkill -f 'gz sim'
pkill -f 'px4'
