# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.test import TestCase


from unittest import TestCase
from app_control.views import full_plan

class TestWithGabor(TestCase):
    def test_train_model(self):
        full_plan()


