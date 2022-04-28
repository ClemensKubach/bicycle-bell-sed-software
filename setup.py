"""Setup script for bicycle-bell-seds-cli"""

import os
from setuptools import setup


readme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'README.md')
with open(readme_path, "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='bicycle-bell-seds-cli',
    version='0.0.1',
    packages=['seds_cli', 'seds_cli.seds_lib', 'seds_cli.seds_lib.data',
              'seds_cli.seds_lib.data.time', 'seds_cli.seds_lib.data.audio',
              'seds_cli.seds_lib.data.configs', 'seds_cli.seds_lib.data.predictions',
              'seds_cli.seds_lib.utils', 'seds_cli.seds_lib.models', 'seds_cli.seds_lib.storage',
              'seds_cli.seds_lib.storage.audio', 'seds_cli.seds_lib.workers',
              'seds_cli.seds_lib.selectors', 'bicycle_bell_seds'],
    url='https://github.com/ClemensKubach/bicycle-bell-sed-software',
    project_urls={
        "Bug Tracker": "https://github.com/ClemensKubach/bicycle-bell-sed-software/issues"
    },
    license='MIT License',
    author='Clemens Kubach',
    author_email='clemens.kubach@gmail.com',
    description='CLI software for single target sound event detection of bicycle bell signals.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        'tensorflow>=2.6.2,<=2.8.0',
        'pyaudio>=0.2.11',
        'fire>=0.4.0',
        'numpy>=1.19.0',
        # 'resampy>=0.2.2',
        'tensorflow_io>=0.21.0,<=0.25.0',
    ],
    extras_require={
        ':python_version < "3.7"': [
            'dataclasses>=0.8',
        ],
    },
    setup_requires=['setuptools_scm'],
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'seds-cli=seds_cli.seds_cli:main',
            'jn-seds-cli=bicycle_bell_seds.jn_seds_cli:main'
        ],
    },

)
