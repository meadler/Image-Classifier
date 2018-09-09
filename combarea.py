#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 15:20:02 2018

@author: Max
"""

def combinedArea(l,c):
    length = len(l)
    combined=0
    total = len(l)+c
    for i in length:
        value = l[i]/total
        combined+= value
    return combined 
        