from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess
import os

current_file_path = os.path.abspath(__file__)
current_folder_path = os.path.dirname(current_file_path)

with open(os.path.join(current_folder_path, 'README.md'), 'r', encoding='utf-8') as fh:
    long_description = fh.read()

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        subprocess.check_call(['pip', 'install', '-r', os.path.join(current_folder_path, 'requirements.txt')])

setup(
    name='face3d_med_reconstruction',
    version='0.0.1',
    include_package_data=True,
    package_data={
        'medface3d': ['data/demo/*.jpg', 'data/demo/*.png', 'data/demo/*.gif'],  # Include all jpg, png, and gif files,
        'medface3d': ['data/sample/*.jpg', 'data/sample/*.png', 'data/sample/*.gif'],  # Include all jpg, png, and gif files
    },
    packages=find_packages(),
    description= "3D face reconstruction using mediapipe 3D Face mesh",
    long_description = long_description,
    long_description_content_type='text/markdown',
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
