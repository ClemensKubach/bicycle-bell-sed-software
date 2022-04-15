"""vorkonfigurierter cli ausführung. definiert callback. Frage ab und leite an seds_cli weiter.
Hier die Möglichkeit mit input() Formular zu starten.
so seds cli aufrufbar machen durch Class SedsCLI mit 2 Methods
python mypy.py 'pathtomodel' --allgemein='ok' production --input=2
"""

# SEDS Folder erstellen wo die seds_cli drin ist. Die kann dann von außerhalb aufgerufen werden
# egal von wo
# dadrüber habe die run scripts speziell für bb on jn
import os

import fire

from seds_cli import seds_constants
from seds_cli import seds_cli


def main():
    seds_cli.main(os.path.join(seds_constants.RES_MODELS_PATH, 'crnn'))


if __name__ == '__main__':
    fire.Fire(main)
