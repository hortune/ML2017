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
    print(colored("Loading model from {}".format('../'),'yellow',attrs=['bold']))
    emotion_classifier = load_model('../last_model')
    emotion_classifier.summary()
    plot_model(emotion_classifier,to_file='model.png')
main()
