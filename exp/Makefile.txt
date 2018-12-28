# coding: utf-8

"""
define generic tasks for the project
"""

init:
    pip install -r requirements.txt

test:
    py.test tests

.PHONY: init test
