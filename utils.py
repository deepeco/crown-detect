#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Created Date: Saturday April 9th 2022
Author: Dmitry Kislov
E-mail: kislov@easydan.com
-----
Last Modified: Saturday, April 9th 2022, 3:00:44 pm
Modified By: Dmitry Kislov
-----
Copyright (c) 2022
"""
import os

def validate_data(root_dir='data'):
    for folder in os.listdir(root_dir):
        im_set = [_ for x, y, z in os.walk(os.path.join(root_dir, folder, 'images')) for _ in z]
        mask_set = [_ for x, y, z in os.walk(os.path.join(root_dir, folder, 'masks')) for _ in z]
        im_set = set(map(lambda x: x.replace('_orig', ''), im_set))
        mask_set = set(mask_set)
        if im_set == mask_set:
            print(f"Folder `{folder}` was validated successfully...")
        else:
            print(f"Data folders are different {folder}:")
            for _ in im_set.union(mask_set):
                if _ in im_set and _ not in mask_set:
                    print(f'File `{_}` exists in images but not in masks.')
                elif _ not in im_set and _ in mask_set:
                    print(f'File `{_}` exists in masks but not in images.')
                else:
                    print(f"File `{_}` file exists in boths.")


    # for dirpath, dirnames, fnames in os.walk(root_dir,):
    #     print(dirpath, dirnames, fnames)

if __name__ == '__main__':
    validate_data()