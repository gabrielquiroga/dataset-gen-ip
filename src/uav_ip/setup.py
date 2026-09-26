from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'uav_ip'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install world files (.sdf + .yaml) into share
        (os.path.join('share', package_name, 'worlds'),
            glob('worlds/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gabrielq',
    maintainer_email='quiroga.m.gabriel@gmail.com',
    description='Instant Policy UAV - Data collection and expert pilot for PX4 SITL',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'expert_pilot = uav_ip.expert_pilot:main',
        ],
    },
)
