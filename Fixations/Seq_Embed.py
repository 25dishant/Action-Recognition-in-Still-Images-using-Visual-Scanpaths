import os
import numpy as np
from collections import OrderedDict
import torch
import pdb

root = os.path.join('~', 'voc_actions', 'Org_traj_image')
# img_id = '2010_006088'
# root = os.path.join(os.path.expanduser(root), img_id)
# observers = ['006','007','008','009','010','011','018','020']
# observer = os.path.join(root, '{}.txt')


def Read_Fixations(img_id, observer_id):
    """
    A function that returns a numpy array of fixation points by an observer with the observer_id passed as argument.

    Signature : This Function is added while working on the Project of Gaze Data Incorporation in the Human-Object Relation Network.

    """
    root = os.path.join('~', 'voc_actions', 'Org_traj_image')
    root = os.path.join(os.path.expanduser(root), img_id)
    # observers = ['006','007','008','009','010','011','018','020']
    observer = os.path.join(root, '{}.txt')
    try: 
        fptr = open(observer.format(observer_id),'r')
    except:
        return np.zeros((2,2))
    lines = fptr.readlines()
    fptr.close()
    for i,line in enumerate(lines):
        lines[i] = list(map(float,line.strip().split(',')))
    fixation_points = np.array(lines)
    return fixation_points


def NearestCenters(fixation_points, objboxcent, remove_duplicates = False):
    """
    A function that returns a list of indices of the object bounding boxes which are nearest to the fixation points. 

    Signature : This Function is added while working on the Project of Gaze Data Incorporation in the Human-Object Relation Network.

    """
    nearest_center_index = []
    for point1 in fixation_points:
        temp = []
        for point2 in objboxcent:
            dist = np.linalg.norm(point1 - point2)
            temp.append(dist)
        i = temp.index(min(temp))
        nearest_center_index.append(i)
        if remove_duplicates:
            nearest_center_index = list(OrderedDict.fromkeys(nearest_center_index))
    return nearest_center_index

def NearestCenters2(fixation_points, objboxcent, remove_duplicates = False):
    """
    A function that returns a list of indices of the object bounding boxes which are nearest to the fixation points. 

    Signature : This Function is added while working on the Project of Gaze Data Incorporation in the Human-Object Relation Network.

    """
    #pdb.set_trace()
    nearest_center_index = []
    for point1 in fixation_points:
        temp = []
        for point2 in objboxcent:
            # dist = np.linalg.norm(point1 - point2)
            dist = torch.dist(point1,point2,p=2)
            temp.append(dist)
        i = temp.index(min(temp))
        nearest_center_index.append(i)
        if remove_duplicates:
            nearest_center_index = list(OrderedDict.fromkeys(nearest_center_index))
    return nearest_center_index


def FixationSequenceEmbedding(box,img_id,observer_id):
    """
    A function that returns only the key object-boxes out of the total object bounding boxes.
    The key object-boxes will be in the order of fixation points.


    Signature : This Function is added while working on the Project of Gaze Data Incorporation in the Human-Object Relation Network.

    """
    fixation_points = Read_Fixations(img_id,observer_id)
    box_centers = np.zeros((len(box),2))
   
    box_centers[:,0] = (box[:,0] + box[:,2])/2
    box_centers[:,1] = (box[:,1] + box[:,3])/2
    nearest_center_index = NearestCenters(fixation_points,box_centers,False)
    keyobjboxes = np.zeros((len(nearest_center_index),4))
    for idx,i in enumerate(nearest_center_index):
        keyobjboxes[idx] = box[i]
    return keyobjboxes


def Sequencer(rois,fixations,device):
    #pdb.set_trace()
    # box_centers = np.zeros((len(rois),2))
    rois = rois.reshape(rois.shape[1],rois.shape[2])
    fixations = fixations.reshape(fixations.shape[1],fixations.shape[2])
    box_centers = torch.zeros((len(rois),2)).to(device)
    box_centers[:,0] = (rois[:,0] + rois[:,2])/2
    box_centers[:,1] = (rois[:,1] + rois[:,3])/2
    nearest_center_index = NearestCenters2(fixations,box_centers,False)
    # keyobjboxes = np.zeros((len(nearest_center_index),4))
    keyobjboxes = torch.zeros((len(nearest_center_index),4)).to(device)
    for idx,i in enumerate(nearest_center_index):
        keyobjboxes[idx] = rois[i]
    return keyobjboxes
    # pass
    