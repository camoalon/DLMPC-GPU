import numpy as np

def convergence_tracker(conv1,conv2):

    csvfile = open('convergence_values','a')
       
    csvfile.write('%f,' %conv1)
    csvfile.write('%f,' %conv2)
    csvfile.write('\n')
    csvfile.close()
