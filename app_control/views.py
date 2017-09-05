# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import json

from django.shortcuts import render
from django.http import JsonResponse

from django.http import HttpResponse

from core.cnn_manager import CNNManager


def index(request):
    return render(request, "index.html", {})


def get_models(request):
    return HttpResponse(json.dumps(CNNManager().get_models()))
