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
import os

from manage import ROOT_DIR
from utils.configurations import load_configurations, get_random_conf

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


@api_view(['GET', 'POST', ])
@csrf_exempt
def full_plan(request):
    try:
        item = 1
        for split_cases in [True, False]:
            for dropout in [0.25, 0.5]:
                for activation_function in ['softmax', 'sigmoid']:
                    for img_size in [(50, 50), (75, 75), (100, 100), (150, 150), (200, 200)]:
                        for sigma in [2, 3, 4, 5, 6]:
                            for theta in [0.3, 45, 90, 135]:
                                for lammbd in [8, 10, 12]:
                                    for gamma in [0.3, 0.5, 0.7, 0.9]:
                                        for psi in [0, 30, 60, 90]:
                                            for nb_filters in [32, 64]:
                                                for kernel_size in [5, 6, 7, 8, 9, 10]:
                                                    for pool_size in [2, 4, 6, 8]:
                                                        for batch_size in [32, 64, 128]:
                                                            try:
                                                                item_path = os.path.join(ROOT_DIR,'cnn_models','item %s.json' % item)
                                                                if os.path.exists(item_path):
                                                                    item += 1
                                                                    break
                                                                params = {}
                                                                params['model_name'] = "item %s" % item
                                                                item += 1
                                                                params['split_cases'] = split_cases
                                                                params['img_rows'] = img_size[0]
                                                                params['img_cols'] = img_size[1]
                                                                params['batch_size'] = batch_size
                                                                params['nb_filters'] = nb_filters
                                                                params['dropout'] = dropout
                                                                params['activation_function'] = activation_function
                                                                params['pool_size'] = pool_size
                                                                params['kernel_size'] = kernel_size
                                                                params['sigma'] = sigma
                                                                params['theta'] = theta
                                                                params['lammbd'] = lammbd
                                                                params['gamma'] = gamma
                                                                params['psi'] = psi

                                                                cnn = CNN(params)
                                                                cnn.train_model(3)
                                                            except Exception as e:
                                                                print e
    except Exception as e:
        print e
        return Response({'msg': e.message}, status=status.HTTP_400_BAD_REQUEST)
    print 'finish all plan'


@api_view(['GET', 'POST', ])
@csrf_exempt
def random_plan(request):
    while True:
        conf = get_random_conf()
        cnn = CNN(conf)
        cnn.train_model(1)
        if 0 in cnn.con_mat_val[-1]:
            continue
        else:
            print 'we find normal model'
            cnn.train_model(3)


@api_view(['GET', 'POST', ])
@csrf_exempt
def good_plan(request):
    try:
        # cnn = CNN({'model_name': 'good_plan_50', 'img_rows': 50, 'img_cols': 50})
        cnn = CNN({'model_name': 'good_plan_50'}, True)
        cnn.train_model(7)
    except Exception as e:
        print e
        return Response({'msg': e.message}, status=status.HTTP_400_BAD_REQUEST)
    print 'finish all plan'
