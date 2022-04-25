"""Module for constants."""

import os

RES_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'res')
RES_LOGS_PATH = os.path.join(RES_PATH, 'logs')
RES_MODELS_PATH = os.path.join(RES_PATH, 'models')
RES_RECORDS_PATH = os.path.join(RES_PATH, 'records')
RES_TESTFILES_PATH = os.path.join(RES_PATH, 'test-files')


def print_res_location():
    """Print base dir of resource folders"""
    print(f'Resource files are located at: {RES_PATH}')


if __name__ == '__main__':
    print_res_location()
