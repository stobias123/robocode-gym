import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import sys

setup(name='robocode-gym',
      version='0.3.0',
      description='Python bindings to Atari games',
      url='https://github.com/openai/atari-py',
      author='OpenAI',
      author_email='info@openai.com',
      license='',
      packages=['gym_robocode'],
)