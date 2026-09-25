# Memoria de Estado del Proyecto

*Este archivo rastrea el progreso del proyecto, las decisiones arquitectónicas tomadas y las tareas pendientes. Debe ser actualizado por el agente conforme se avance.*

## ✅ Logros Completados
- **Setup Base:** Instalación exitosa de Ubuntu 24.04 (WSL2), ROS 2 Jazzy, Gazebo Harmonic y PX4 SITL.
- **Comunicación:** Enlace DDS establecido mediante Micro XRCE-DDS Agent.
- **Piloto Experto (v1):** Script Python creado (`expert_pilot.py`) que publica `TrajectorySetpoint` sincronizado con el timestamp de PX4, cruzando exitosamente aros fijos.
- **Integración de Sensores:** `ros_gz_bridge` configurado correctamente para extraer `PointCloudPacked` hacia `sensor_msgs/PointCloud2`.
- **Validación de Datos:** Grabación exitosa de rosbags (`.mcap` de 6+ GiB) conteniendo Odometría, Acciones y Nubes de Puntos, validados visualmente con Foxglove Studio.

## 🚧 Tareas Activas / Inmediatas
1. **Script de Aleatorización:** Crear el script de Gazebo/ROS que haga spawn de los aros en nuevas posiciones aleatorias de forma programática.
2. **Pipeline de Extracción (Dataset):**
   - Implementar `dataset_extractor.py` usando `rosbags`.
   - Escribir la función de conversión de `PointCloud2` a matriz de Numpy.

## 🎯 Deuda Técnica / Próximos Pasos (Instant Policy Compliance)
- **Segmentación:** Integrar un plugin de segmentación semántica en Gazebo O crear una heurística espacial en Python para borrar el suelo de la nube de puntos.
- **Marco Egocéntrico:** Modificar `dataset_extractor.py` para transformar la Odometría global a desplazamientos relativos.
- **Keyframing:** Desarrollar la heurística temporal para no guardar el rosbag a 30Hz continuos, sino extraer ~10 estados críticos por episodio.