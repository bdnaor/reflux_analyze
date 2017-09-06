# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import json

from django.shortcuts import render

from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view

from core.cnn import CNN
from core.cnn_manager import CNNManager
from rest_framework import status
from rest_framework.response import Response
cnn_manager = CNNManager()


def index(request):
    return render(request, "index.html", {})


def get_models(request):
    return HttpResponse(json.dumps(cnn_manager.get_models()))


@api_view(['GET', 'POST', ])
@csrf_exempt
def add_model(request):
    model_name = request.POST['model_name']
    img_rows = int(request.POST['img_rows'])
    img_cols = int(request.POST['img_cols'])
    epoch = int(request.POST['epoch'])
    kernel_size = int(request.POST['kernel_size'])
    pool_size = int(request.POST['pool_size'])
    cnn = CNN(img_rows=img_rows, img_cols=img_cols, epoch=epoch, pool_size=pool_size, kernel_size=kernel_size)
    success, msg = cnn_manager.add_model(cnn, model_name)
    if success:
        return Response({'msg': msg}, status=status.HTTP_200_OK)
    else:
        return Response({'msg': msg}, status=status.HTTP_400_BAD_REQUEST)
