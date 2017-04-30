#!/usr/bin/env python
# -- coding: utf-8 --

import os
from termcolor import colored, cprint
import argparse
from keras.utils import plot_model
from keras.models import load_model

base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
#exp_dir = os.path.join(base_dir,'exp')

def main():
    """
    parser = argparse.ArgumentParser(prog='plot_model.py',
            description='Plot the model.')
    parser.add_argument('--model',type=str,default='simple',choices=['simple','easy','strong'],
            metavar='<model>')
    parser.add_argument('--epoch',type=int,metavar='<#epoch>',default=80)
    parser.add_argument('--batch',type=int,metavar='<batch_size>',default=64)
    parser.add_argument('--idx',type=int,metavar='<suffix>',required=True)
    args = parser.parse_args()
    """
    print(colored("Loading model from {}".format('../'),'yellow',attrs=['bold']))
    
    emotion_classifier = load_model('../last_model')
    emotion_classifier.summary()
    plot_model(emotion_classifier,to_file='model.png')
main()
