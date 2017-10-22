#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 15:04:28 2017

@author: elef
"""

from __future__ import division

import numpy as np
import vtk

from numpy.testing import (assert_, assert_equal, assert_array_equal,
                           assert_array_almost_equal, assert_almost_equal,
                           run_module_suite)

from dipy.data import get_data
from dipy.reconst.dti import TensorModel
from dipy.reconst.shm import CsaOdfModel
from dipy.sims.phantom import orbital_phantom
from dipy.core.gradients import gradient_table


fimg, fbvals, fbvecs = get_data('small_64D')
bvals = np.load(fbvals)
bvecs = np.load(fbvecs)
bvecs[np.isnan(bvecs)] = 0

gtab = gradient_table(bvals, bvecs)


def f1(t):
    x = np.linspace(-1, 1, len(t))
    y = np.linspace(-1, 1, len(t))
    z = np.zeros(x.shape)
    return x, y, z


def f2(t):
    x = np.linspace(-1, 1, len(t))
    y = -np.linspace(-1, 1, len(t))
    z = np.zeros(x.shape)
    return x, y, z

class Node(object):
    def __init__(self, position, value=0):
        self.position = position
        self.value = value
        self.previous = []
        self.next = []
"""
class State_action(object):
    def __init__(self, direction, position, value=0, index=0):
        self.direction = direction
        self.position = position
        self.value = value
        self.index = index
"""

class Connection(object):
    def __init__(self, index, direction):
        self.index = index
        self.direction = direction

class Seed(object):
    def __init__(self, position, index=0):
        self.position = position
        self.index = index
        self.track1 = []


if __name__ == "__main__":

    N = 200
    S0 = 100
    N_angles = 32
    N_radii = 20
    vol_shape = (100, 100, 100)
    origin = (50, 50, 50)
    scale = (30, 30, 30)
    t = np.linspace(0, 2 * np.pi, N)
    angles = np.linspace(0, 2 * np.pi, N_angles)
    radii = np.linspace(0.2, 2, N_radii)
    """
    vol1 = orbital_phantom(gtab,
                           func=f1,
                           t=t,
                           datashape=vol_shape + (len(bvals),),
                           origin=origin,
                           scale=scale,
                           angles=angles,
                           radii=radii,
                           S0=S0)

    vol2 = orbital_phantom(gtab,
                           func=f2,
                           t=t,
                           datashape=vol_shape + (len(bvals),),
                           origin=origin,
                           scale=scale,
                           angles=angles,
                           radii=radii,
                           S0=S0)

    vol = vol1 + vol2
    """
    vol = np.load('vol.npy')
    tensor_model = TensorModel(gtab)
    tensor_fit = tensor_model.fit(vol)
    FA = tensor_fit.fa
    # print vol
    FA[np.isnan(FA)] = 0
    # 686 -> expected FA given diffusivities of [1500, 400, 400]
    l1, l2, l3 = 1500e-6, 400e-6, 400e-6
    expected_fa = (np.sqrt(0.5) *
                   np.sqrt((l1 - l2)**2 + (l2-l3)**2 + (l3-l1)**2) /
                   np.sqrt(l1**2 + l2**2 + l3**2))

    mask = FA > 0.1

    csa_model = CsaOdfModel(gtab, 8)
    from dipy.direction import peaks_from_model
    from dipy.data import get_sphere
    from scipy.spatial import distance
    from numpy.linalg import norm

    pam = peaks_from_model(csa_model, vol, get_sphere('repulsion724'),
                           relative_peak_threshold=.5,
                           min_separation_angle=25,
                           mask=mask,
                           parallel=True)

#    seeds = utils.seeds_from_mask(seed_mask, density=[2, 2, 2], affine=affine)

    non_zero = np.where(FA !=0 )
    n = np.array(non_zero)
    n = np.transpose(n)

    seed = n[1]
    dim1 = np.linspace(n[1][0]-1,n[1][0]+1)
    dim2 = np.linspace(n[1][1]-1,n[1][1]+1)
    dim3 = np.linspace(n[1][2]-1,n[1][2]+1)
    streamlines = []
#    streamline = seed
    Points = vtk.vtkPoints()
    vertices = vtk.vtkCellArray()
    seed_nodes = []
    graph = []
    seeds = []
    nodes = []

    def Td_learning(graph, onetrack, reward_positive=1, reward_negative=-1):
        for


    def return_streamline(seed,dirs,track_point,graph):
        streamline = seed
        node_onetrack = []
        if len(graph.shape) == 1:
            index_c = 0
            node_onetrack = seed
        if len(graph.shape) != 1:
            norm2 = norm(graph-seed,axis=1,ord=2)
            if norm2.min() < 1:
                index_c = np.argmin(norm2)
                node_onetrack = graph[index_c]
            else:
                index_c = graph.shape[0]
                graph = np.vstack((graph,seed))
                node_onetrack = seed

        seed_onetrack = Seed(seed, index_c)
        np.append(seed_onetrack.track1, index_c)
        #seeds.append(seed_onetrack)

        #state_action = State_action(dirs[int(np.round(track_point[0])),int(np.round(track_point[1])),int(np.round(track_point[2])),0,:], track_point, 0)
        #graph.append(node)
        """
        id = Points.InsertNextPoint(seed[0],seed[1],seed[2])
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(id)
        """
        while(FA[int(np.round(track_point[0])),int(np.round(track_point[1])),int(np.round(track_point[2]))] != 0):
            direction = dirs[int(np.round(track_point[0])),int(np.round(track_point[1])),int(np.round(track_point[2])),0,:]
            track_point = track_point + direction
            if len(graph.shape) == 1:
                norm2 = norm(graph-track_point)
            else:
                norm2 = norm(graph-track_point,axis=1,ord=2)
            if norm2.min() < 1:
                index_t = np.argmin(norm2)
                if not np.any(seed_onetrack.track1 == index_t):
                    seed_onetrack.track1.append(index_t)
                #print(node_onetrack)
                    if len(graph.shape) == 1:
                        node_onetrack = np.vstack((node_onetrack,graph))
                    else:
                        node_onetrack = np.vstack((node_onetrack,graph[index_t]))
            else:
                index_t = graph.shape[0]
                graph = np.vstack((graph,track_point))
                seed_onetrack.track1.append(index_t)
                node_onetrack = np.vstack((node_onetrack,track_point))


            #node = Node(direction, track_point,0)
            #graph.append(node)
            streamline = np.vstack((streamline,track_point))
            """
            id = Points.InsertNextPoint(track_point[0],track_point[1],track_point[2])
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(id)
            """
        return streamline,graph,seed_onetrack, node_onetrack

    seed = n[1500]
    graph = seed
    seed_nodes = seed
    #seed_onetrack = Seed(seed, 0)
    #seeds.append(seed_onetrack)
    stremlines_onetrack, graph_onetrack,seed_onetrack, node_onetrack1 = return_streamline(seed,pam.peak_dirs,seed,graph)
    streamlines.append(stremlines_onetrack)
    nodes.append(node_onetrack1)
    seeds.append(seed_onetrack)
    graph = graph_onetrack

    for i in range(500):#len(n)):
        seed = n[i+1501]
        stremlines_onetrack, graph_onetrack,seed_onetrack, node_onetrack1 = return_streamline(seed,pam.peak_dirs,seed,graph)
        #streamlines.append(return_streamline(seed,pam.peak_dirs,seed,graph))
        streamlines.append(stremlines_onetrack)
        nodes.append(node_onetrack1)
        seeds.append(seed_onetrack)
        graph = graph_onetrack

    from dipy.viz import actor, window
    from dipy.tracking.local import LocalTracking
    from dipy.viz import fvtk
    from dipy.viz.colormap import line_colors

    color = line_colors(streamlines)

    if fvtk.have_vtk:
        streamlines_actor = actor.line(streamlines)

        r = window.Renderer()
        r.background((1,1,1))
        r.add(streamlines_actor)
        window.show(r)
        r.clear()
        node_actor = actor.streamtube([nodes[100]])
        colors = np.random.rand(graph.shape[0],3)
        point_actor = fvtk.point(graph,colors)
        r.add(point_actor)
        r.add(node_actor)
        window.show(r)
        #r.rm(streamlines_actor)


    """
    # create a rendering window and renderer
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)

    # create a renderwindowinteractor
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    ren.SetBackground(1,1,1)
    """


    #id = Points.InsertNextPoint(p)

    #vertices.InsertNextCell(1)
    #vertices.InsertCellPoint(id)
    """
    point = vtk.vtkPolyData()
    point.SetPoints(Points)
    point.SetVerts(vertices)
    # mapper
    mapper = vtk.vtkPolyDataMapper()
    #mapper.SetInputConnection(src.GetOutputPort())
    mapper.SetInputData(point)

    # actor
    actor = vtk.vtkActor()
    actor.GetProperty().SetColor(1,0,0)
    actor.GetProperty().SetPointSize(10)
    actor.GetProperty().SetRenderPointsAsSpheres(1)
    actor.SetMapper(mapper)

    # assign actor to the renderer
    ren.AddActor(actor)

    # enable user interface interactor
    iren.Initialize()
    renWin.Render()
    iren.Start()
    """



    """
    ren = window.Renderer()

    actor_slicer = actor.slicer(FA, interpolation='nearest')
    ren.add(actor_slicer)

    tmp = pam.peak_values
    # tmp[FA<0.2] = 0
    peak_slicer = actor.peak_slicer(pam.peak_dirs, tmp,
                                    colors=(1., 0., 0.), lod=False)
    ren.add(peak_slicer)
    #from dipy.reconst.shm import sh_to_sf
    #odf = sh_to_sf(pam.shm_coeff, get_sphere('repulsion724'), 8)
    #odf_slicer = actor.odf_slicer(odf, sphere=get_sphere('repulsion724'))
    window.show(ren)
    """
