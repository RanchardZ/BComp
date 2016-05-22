#!/bin/sh
cp -n ../library/bbcomp.dll      . 2>/dev/null || :
cp -n ../library/libbbcomp.so    . 2>/dev/null || :
cp -n ../library/libbbcomp.dylib . 2>/dev/null || :
python client_zhenghe.py
