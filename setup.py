from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess
import os

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        current_file_path = os.path.abspath(__file__)
        current_folder_path = os.path.dirname(current_file_path)
        install.run(self)
        subprocess.check_call(['pip', 'install', '-r', os.path.join(current_folder_path, 'requirements.txt')])

setup(
    name='med_face_reconstruction',
    version='0.1.0',
    packages=find_packages(),
    description= "3D face reconstruction using mediapipe 3D Face mesh",
    install_requires=[
        # List other pip dependencies here if any
    ],
    cmdclass={
        'install': PostInstallCommand,
    },
    python_requires='>=3.7',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ]
)
