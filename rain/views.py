from django.shortcuts import render
from django.http import HttpResponse
from django.template.context_processors import request

def index(request):
    return HttpResponse("Hello world")