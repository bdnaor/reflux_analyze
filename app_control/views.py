# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import json
from threading import Thread

from django.shortcuts import render

from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view

from core.cnn import CNN
from core.cnn_manager import CNNManager
from rest_framework import status
from rest_framework.response import Response
from multiprocessing import Process

cnn_manager = CNNManager()




def index(request):
    return render(request, "index.html", {})


def get_models(request):
    return HttpResponse(json.dumps(cnn_manager.get_models()))


@api_view(['GET', 'POST', ])
@csrf_exempt
def add_model(request):
    model_name = request.POST['model_name']
    # img_rows = int(request.POST['img_rows'])
    # img_cols = int(request.POST['img_cols'])
    # epoch = int(request.POST['epoch'])
    # kernel_size = int(request.POST['kernel_size'])
    # pool_size = int(request.POST['pool_size'])
    cnn = CNN(request.POST)
    success, msg = cnn_manager.add_model(cnn)
    if success:
        return Response({'msg': msg}, status=status.HTTP_200_OK)
    else:
        return Response({'msg': msg}, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET', 'POST', ])
@csrf_exempt
def predict_images(request, *args, **kwargs):
    predictions = {}
    try:
        if request.method == 'POST':
            model_name = request.POST['model_name']
            cnn = cnn_manager.models[model_name]
            for image in request.FILES.keys():
                predictions[image] = cnn.predict(request.FILES[image])
    except Exception as e:
        return Response({'msg': e.message}, status=status.HTTP_400_BAD_REQUEST)

    return Response({'predictions': predictions}, status=status.HTTP_200_OK)


@api_view(['GET', 'POST', ])
@csrf_exempt
def start_train(request):
    try:
        if request.method == 'POST':
            model_name = request.POST['model_name']
            epoch = int(request.POST['epoch'])
            cnn = cnn_manager.models[model_name]
            # p = Process(target=cnn.train_model, args=(epoch,))
            # p.start()
            # thread = Thread(target=cnn.train_model, args=(epoch,))
            # thread.start()
            cnn.train_model(epoch)
        return Response({'msg': 'ok'}, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({'msg': e.message}, status=status.HTTP_400_BAD_REQUEST)


# class FileFieldForm(forms.Form):
#     file_field = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': True}))
#
#
# @api_view(['GET', 'POST', ])
# @csrf_exempt
# class FileFieldView(FormView):
#     form_class = FileFieldForm
#     template_name = 'index.html'
#     success_url = '/predict_images'  # Replace with your URL or reverse().
#
#     def post(self, request, *args, **kwargs):
#         form_class = self.get_form_class()
#         form = self.get_form(form_class)
#         files = request.FILES.getlist('file_field')
#         if form.is_valid():
#             for f in files:
#                 pass
#             return self.form_valid(form)
#         else:
#             return self.form_invalid(form)

