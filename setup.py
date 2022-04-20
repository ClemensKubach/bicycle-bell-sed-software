from setuptools import setup

setup(
    name='bicycle_bell_seds_cli',
    version='0.0.1',
    packages=['seds_cli', 'seds_cli.seds_lib', 'seds_cli.seds_lib.data',
              'seds_cli.seds_lib.data.time', 'seds_cli.seds_lib.data.audio',
              'seds_cli.seds_lib.data.configs', 'seds_cli.seds_lib.data.predictions',
              'seds_cli.seds_lib.utils', 'seds_cli.seds_lib.models', 'seds_cli.seds_lib.storage',
              'seds_cli.seds_lib.storage.audio', 'seds_cli.seds_lib.workers',
              'seds_cli.seds_lib.selectors', 'bicycle_bell_seds'],
    url='https://github.com/ClemensKubach/bicycle-bell-sed-software',
    license='MIT License',
    author='Clemens Kubach',
    author_email='clemens.kubach@gmail.com',
    description='',
    install_requires=[
        'tensorflow>=2.6.2',
        'tensorflow_io>=0.21.0',
        'numpy',
        'pyaudio>=0.2.11',
        'fire>=0.4.0'
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
            'seds_bb_desktop=bicycle_bell_seds.run_desktop_bicycle_bell_seds:main',
            'seds_bb_jn=bicycle_bell_seds.run_jn_bicycle_bell_seds:main',
            'seds_cli=seds_cli.seds_cli:main'
        ],
    },
)
