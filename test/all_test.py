import os


for file in os.listdir('./'):
    if file.startswith('test') and file.endswith('.py'):
        print('\n*********************testing '+ file + ' *********************\n') 
        os.system('python '+file)