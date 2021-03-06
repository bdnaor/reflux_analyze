import os
import json

from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from rest_framework import status
from rest_framework.response import Response
from django.http import HttpResponse
from manage import ROOT_DIR
from core.cnn import CNN
from core.cnn_manager import CNNManager
from utils.configurations import get_random_conf

from PIL import Image

cnn_manager = CNNManager()


def index(request):
    return render(request, "index.html", {})


def get_models(request):
    return HttpResponse(json.dumps(cnn_manager.get_models()))


@api_view(['GET', 'POST', ])
@csrf_exempt
def add_model(request):
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
def predict_random_frame(request, *args, **kwargs):
    try:
        model_name = request.POST['model_name']
        data = cnn_manager.get_random_frame(model_name)
        try:
            return Response(data, status=status.HTTP_200_OK)
            # return HttpResponse(data['img'], content_type="image/png")
        except IOError:
            red = Image.new('RGBA', (1, 1), (255, 0, 0, 0))
            response = HttpResponse(content_type="image/png", status=status.HTTP_200_OK)
            red.save(response, "PNG")
            return response
    except Exception as e:
        return Response({'msg': e.message}, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET', 'POST', ])
@csrf_exempt
def start_train(request):
    try:
        if request.method == 'POST':
            model_name = request.POST['model_name']
            epoch = int(request.POST['epoch'])
            cnn = cnn_manager.models[model_name]
            cnn.train_model(epoch)
        return Response({'msg': 'ok'}, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({'msg': e.message}, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET', 'POST', ])
@csrf_exempt
def full_plan(request):
    try:
        item = 1
        for split_cases in ['True', 'False']:
            for dropout in [0.25, 0.5]:
                for activation_function in ['softmax', 'sigmoid']:
                    for img_size in [(75, 75), (50, 50)]:
                        for nb_filters in [32, 64]:
                            for kernel_size in [5, 6, 7, 8, 9, 10]:
                                for pool_size in [2, 4, 6, 8]:
                                    for batch_size in [32, 64, 128]:
                                        for sigma in [180, 90, 30]:
                                            for theta in [45, 90, 135]:
                                                for lammbd in [45, 90, 135]:
                                                    for gamma in [0.3, 0.5, 0.7, 0.9]:
                                                        for psi in [0.2, 0.5, 0.8]:
                                                            try:
                                                                item_path = os.path.join(ROOT_DIR,'cnn_models','item%s.json' % item)
                                                                if os.path.exists(item_path):
                                                                    item += 1
                                                                    break
                                                                params = {}
                                                                params['model_name'] = "item%s" % item
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
                                                                cnn.train_model(150)
                                                            except Exception as e:
                                                                print e
    except Exception as e:
        print e
        return Response({'msg': e.message}, status=status.HTTP_400_BAD_REQUEST)
    print 'finish all plan'
    return Response({}, status=status.HTTP_200_OK)


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
            cnn.train_model(10)


@api_view(['GET', 'POST', ])
@csrf_exempt
def good_plan(request):
    try:
        # cnn = CNN({'model_name': 'good_plan_50', 'img_rows': 50, 'img_cols': 50})
        cnn = CNN({'model_name': 'good_plan_50'}, True)
        cnn.train_model(40)
    except Exception as e:
        print e
        return Response({'msg': e.message}, status=status.HTTP_400_BAD_REQUEST)
    print 'finish good plan'
    return Response({}, status=status.HTTP_200_OK)
