# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import os


for file in os.listdir('./'):
    if file.startswith('test') and file.endswith('.py'):
        print('\n*********************testing '+ file + ' *********************\n') 
        os.system('python '+file)