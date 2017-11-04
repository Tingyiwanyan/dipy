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
        self.track2 = []
        self.nodes = []

class Seed_node_graph(object):
    def __init__(self, graph, value):
        self.graph = graph
        self.value = value


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
    seeds_nodes_graph = []
    graph = []
    seeds = []
    seeds_index = []
    nodes = []

    def ev_learning(graph, onetrack1, onetrack2, value, reward_positive=100, reward_negative=-100, positive1=True, positive2=True, alfa=0.95, gamma=0.8):
        #value = np.zeros(graph.shape[0])
        #print(len(value))
        #print(onetrack2)
        print("value")
        #print(value)
        if positive1 == True:
            reward1 = reward_positive
        else:
            reward1 = reward_negative
        for i in range(len(onetrack1)):
            if i == 0:
                value[int(onetrack1[len(onetrack1)-(i+1)])] = reward1
            else:
                #print(i)
                #print(value[int(onetrack1[len(onetrack1)-i])])
                value[int(onetrack1[len(onetrack1)-(i+1)])] = value[int(onetrack1[len(onetrack1)-(i+1)])] + alfa*(value[int(onetrack1[len(onetrack1)-i])]-value[int(onetrack1[len(onetrack1)-(i+1)])])
                #print(value[int(onetrack1[len(onetrack1)-(i+1)])])

        if positive2 == True:
            reward2 = reward_positive
        else:
            reward2 = reward_negative
        for i in range(len(onetrack2)):
            if i == 0:
                value[int(onetrack2[len(onetrack2)-(i+1)])] = reward2
            else:
                value[int(onetrack2[len(onetrack2)-(i+1)])] = value[int(onetrack2[len(onetrack2)-(i+1)])] + alfa*(value[int(onetrack2[len(onetrack2)-i])]-value[int(onetrack2[len(onetrack2)-(i+1)])])

        return value


    def find_track_point(dirs,track_point,onetrack,graph,value,index,with_stand, direction=True):
        direction1 = dirs[int(np.round(track_point[0])),int(np.round(track_point[1])),int(np.round(track_point[2])),index,:]
        if direction == True:
            track_point_test = track_point + direction1
        else:
            track_point_test = track_point - direction1
        if len(graph.shape) == 1:
            norm2 = norm(graph-track_point_test)
        else:
            norm2 = norm(graph-track_point_test,axis=1,ord=2)
        if norm2.min() < 1:
            index_t = np.argmin(norm2)
            if value[int(index_t)] > with_stand or value[int(index_t)] == with_stand:
                if not np.any(onetrack == index_t):
                    #seed_onetrack.track1 = np.append(seed_onetrack.track1,index_t)
                    onetrack = np.append(onetrack,index_t)
                    return 0, onetrack, index_t, graph, value
            #print(node_onetrack)
                """
                if len(graph.shape) == 1:
                    node_onetrack = np.vstack((node_onetrack,graph))
                else:
                    node_onetrack = np.vstack((node_onetrack,graph[index_t]))
                """
            #if value[index_t] < with_stand:
            #    return [], onetrack, index_t, graph, value
            """
            else:
                index = index + 1
                if dirs[int(np.round(track_point[0])),int(np.round(track_point[1])),int(np.round(track_point[2])),index,:].max() !=0:
                    track_point_test2 = track_point + dirs[int(np.round(track_point[0])),int(np.round(track_point[1])),int(np.round(track_point[2])),index,:]
                    return find_track_point(dirs, track_point,onetrack,graph,value,index,with_stand)
                else:
                    #onetrack = np.append(onetrack,index_t)
                    return 0, onetrack, index_t, graph, value
            """
        else:
            if len(graph.shape) == 1:
                index_t = 1
            else:
                index_t = graph.shape[0]
            graph = np.vstack((graph,track_point_test))
            value = np.append(value,0.0)
            onetrack = np.append(onetrack, index_t)
            #node_onetrack = np.vstack((node_onetrack,track_point))
            return track_point_test, onetrack, index_t, graph, value

    def find_track_point2(dirs, track_point, graph, value, direc=True):
        if direc == True:
            track_point_test = track_point + dirs
        else:
            track_point_test = track_point - dirs
        if len(graph.shape) == 1:
            norm2 = norm(graph-track_point_test)
        else:
            norm2 = norm(graph-track_point_test,axis=1,ord=2)
        if norm2.min() < 1:
            index_t = np.argmin(norm2)
            return value[index_t], 1
        else:
            return 0.0, 0



    def return_streamline(seed,dirs,track_point,graph,value,with_stand=-3):
        streamline = seed
        node_onetrack = []
        decision = 1
        if len(graph.shape) == 1:
            index_c = 0
            node_onetrack = seed
        if len(graph.shape) != 1:
            norm2 = norm(graph-seed,axis=1,ord=2)
            if norm2.min() < 1:
                index_c = np.argmin(norm2)
                #if value[index_c] < with_stand:
                #    return []
                node_onetrack = graph[index_c]
            else:
                index_c = graph.shape[0]
                graph = np.vstack((graph,seed))
                value = np.append(value,0.0)
                node_onetrack = seed

        seed_onetrack = Seed(seed, index_c)
        seed_onetrack.track1 = np.append(seed_onetrack.track1, index_c)
        seed_onetrack.track2 = np.append(seed_onetrack.track2, index_c)
        #seeds.append(seed_onetrack)

        #state_action = State_action(dirs[int(np.round(track_point[0])),int(np.round(track_point[1])),int(np.round(track_point[2])),0,:], track_point, 0)
        #graph.append(node)
        """
        id = Points.InsertNextPoint(seed[0],seed[1],seed[2])
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(id)
        """
        while(FA[int(np.round(track_point[0])),int(np.round(track_point[1])),int(np.round(track_point[2]))] != 0 ):
        #node = Node(direction, track_point,0)
        #graph.append(node)
            #index = 0
            with_stand = -5
            index_inside = 0
            value_single = -500
            for i in range(5):
                dir_sub = dirs[int(np.round(track_point[0])),int(np.round(track_point[1])),int(np.round(track_point[2])),i,:]
                if dir_sub.all() == True:
                    value_single_test, out_flag = find_track_point2(dir_sub, track_point, graph, value, direc=True)
                    print("value_single_test")
                    print(value_single_test)
                    if value_single_test > value_single:
                        index_inside = i
                        value_single = value_single_test
            track_point = track_point + dirs[int(np.round(track_point[0])),int(np.round(track_point[1])),int(np.round(track_point[2])),index_inside,:]
            if len(graph.shape) == 1:
                norm2 = norm(graph-track_point)
            else:
                norm2 = norm(graph-track_point,axis=1,ord=2)
            if norm2.min() < 1:
                index_t = np.argmin(norm2)
                if value[int(index_t)] > with_stand or value[int(index_t)] == with_stand:
                    if not np.any(seed_onetrack.track1 == index_t):
                        #seed_onetrack.track1 = np.append(seed_onetrack.track1,index_t)
                        seed_onetrack.track1 = np.append(seed_onetrack.track1,index_t)
                else:
                    if not np.any(seed_onetrack.track1 == index_t):
                        #seed_onetrack.track1 = np.append(seed_onetrack.track1,index_t)
                        seed_onetrack.track1 = np.append(seed_onetrack.track1,index_t)
                        decision = 0
            else:
                if len(graph.shape) == 1:
                    index_t = 1
                else:
                    index_t = graph.shape[0]
                graph = np.vstack((graph,track_point))
                value = np.append(value,0.0)
                seed_onetrack.track1 = np.append(seed_onetrack.track1, index_t)



            #track_point,seed_onetrack.track1,index_t,graph,value=find_track_point(dirs,track_point,seed_onetrack.track1,graph,value,index,with_stand)#, direction=True)
            streamline = np.vstack((streamline,track_point))

        track_point = seed
        while(FA[int(np.round(track_point[0])),int(np.round(track_point[1])),int(np.round(track_point[2]))] != 0):
            """
            direction = dirs[int(np.round(track_point[0])),int(np.round(track_point[1])),int(np.round(track_point[2])),0,:]
            track_point = track_point - direction
            if len(graph.shape) == 1:
                norm2 = norm(graph-track_point)
            else:
                norm2 = norm(graph-track_point,axis=1,ord=2)
            if norm2.min() < 1:
                index_t = np.argmin(norm2)
                if not np.any(seed_onetrack.track2 == index_t):
                    seed_onetrack.track2 = np.append(seed_onetrack.track2,index_t)
                #print(node_onetrack)
                    if len(graph.shape) == 1:
                        node_onetrack = np.vstack((node_onetrack,graph))
                    else:
                        node_onetrack = np.vstack((node_onetrack,graph[index_t]))
            else:
                if len(graph.shape) == 1:
                    index_t = 1
                else:
                    index_t = graph.shape[0]
                graph = np.vstack((graph,track_point))
                value = np.append(value,0.0)
                seed_onetrack.track2 = np.append(seed_onetrack.track2, index_t)
                node_onetrack = np.vstack((node_onetrack,track_point))

        #node = Node(direction, track_point,0)
        #graph.append(node)
            """
            with_stand = -5
            index_inside = 0
            value_single = -500
            for i in range(5):
                dir_sub = dirs[int(np.round(track_point[0])),int(np.round(track_point[1])),int(np.round(track_point[2])),i,:]
                if dir_sub.all() == True:
                    value_single_test, out_flag = find_track_point2(dir_sub, track_point, graph, value, direc=False)
                    print("value_single_test")
                    print(value_single_test)
                    if value_single_test > value_single:
                        index_inside = i
                        value_single = value_single_test
            track_point = track_point - dirs[int(np.round(track_point[0])),int(np.round(track_point[1])),int(np.round(track_point[2])),index_inside,:]
            if len(graph.shape) == 1:
                norm2 = norm(graph-track_point)
            else:
                norm2 = norm(graph-track_point,axis=1,ord=2)
            if norm2.min() < 1:
                index_t = np.argmin(norm2)
                if value[int(index_t)] > with_stand or value[int(index_t)] == with_stand:
                    if not np.any(seed_onetrack.track2 == index_t):
                        #seed_onetrack.track1 = np.append(seed_onetrack.track1,index_t)
                        seed_onetrack.track2 = np.append(seed_onetrack.track2,index_t)
                else:
                    if not np.any(seed_onetrack.track2 == index_t):
                        #seed_onetrack.track1 = np.append(seed_onetrack.track1,index_t)
                        seed_onetrack.track2 = np.append(seed_onetrack.track2,index_t)
                        decision = 0
            else:
                if len(graph.shape) == 1:
                    index_t = 1
                else:
                    index_t = graph.shape[0]
                graph = np.vstack((graph,track_point))
                value = np.append(value,0.0)
                seed_onetrack.track2 = np.append(seed_onetrack.track2, index_t)



            #track_point,seed_onetrack.track1,index_t,graph,value=find_track_point(dirs,track_point,seed_onetrack.track1,graph,value,index,with_stand)#, direction=True)
            streamline = np.vstack((streamline,track_point))
        #value = []
        value = ev_learning(graph, seed_onetrack.track1, seed_onetrack.track2,value)
        """
        id = Points.InsertNextPoint(track_point[0],track_point[1],track_point[2])
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(id)
        """
        return streamline,graph,seed_onetrack, node_onetrack, value

    seed = n[1500]
    graph = seed
    value = [0.0]
    seed_nodes = seed
    #seed_onetrack = Seed(seed, 0)
    #seeds.append(seed_onetrack)
    stremlines_onetrack, graph_onetrack,seed_onetrack, node_onetrack1,value = return_streamline(seed,pam.peak_dirs,seed,graph,value)
    streamlines.append(stremlines_onetrack)
    seed_onetrack.nodes = node_onetrack1
    seed_onetrack.index = 0
    #nodes.append(node_onetrack1)
    seeds.append(seed_onetrack)
    one_seed_node_graph = Seed_node_graph(graph_onetrack, value)
    seeds_nodes_graph.append(one_seed_node_graph)
    #graph = graph_onetrack

    for i in range(20):#len(n)):
        seed = n[i+1501]
        if len(seed_nodes.shape) == 1:
            norm2 = norm(seed_nodes-seed)
        if len(seed_nodes.shape) != 1:
            norm2 = norm(seed_nodes-seed,axis=1,ord=2)
        if norm2.min() < 1:
            index_c = np.argmin(norm2)
            stremlines_onetrack, graph_onetrack,seed_onetrack, node_onetrack1,value = return_streamline(seed,pam.peak_dirs,seed,seeds_nodes_graph[index_c].graph,seeds_nodes_graph[index_c].value)
            print("streamlines")
            print(streamlines)
            seeds_nodes_graph[index_c].graph = graph_onetrack
            seeds_nodes_graph[index_c].value = value
            seed_onetrack.index = index_c
            seed_onetrack.nodes = node_onetrack1
            #seeds[index_c] = seed_onetrack
            seeds.append(seed_onetrack)
        else:
            index_c = seed_nodes.shape[0]
            seed_nodes = np.vstack((seed_nodes,seed))
            graph = seed
            value = [0.0]
            stremlines_onetrack, graph_onetrack,seed_onetrack, node_onetrack1, value = return_streamline(seed,pam.peak_dirs,seed,graph,value)
            one_seed_node_graph = Seed_node_graph(graph_onetrack, value)
            seeds_nodes_graph.append(one_seed_node_graph)
            seed_onetrack.nodes = node_onetrack1
            seed_onetrack.index = index_c
            seeds.append(seed_onetrack)

        #sstremlines_onetrack, graph_onetrack,seed_onetrack, node_onetrack1 = return_streamline(seed,pam.peak_dirs,seed,graph)
        #streamlines.append(return_streamline(seed,pam.peak_dirs,seed,graph))
        streamlines.append(stremlines_onetrack)
        #nodes.append(node_onetrack1)
        #seeds.append(seed_onetrack)
        #graph = graph_onetrack

    from dipy.viz import actor, window
    from dipy.tracking.local import LocalTracking
    from dipy.viz import fvtk
    from dipy.viz.colormap import line_colors

    color = line_colors(streamlines)

    if fvtk.have_vtk:
        streamlines_actor = actor.line(streamlines)

        r = window.Renderer()
        fvtk.clear(r)
        r.background((1,1,1))
        fvtk.add(r, streamlines_actor)
        window.show(r)
        for i in range(len(seeds_nodes_graph)):
            #r.clear()
            #node_actor = actor.streamtube([nodes[100]])
            if len(seeds_nodes_graph[i].graph.shape) == 1:
                #colors = np.random.rand(1,3)
                colors = np.array((1,1 - seeds_nodes_graph[i].value[0]/100,1 - seeds_nodes_graph[i].value[0]/100))
                point_actor = fvtk.point(seeds_nodes_graph[i].graph[None,:],colors, point_radius=0.3)
            else:
                #colors = np.random.rand(seeds_nodes_graph[i].graph.shape[0],3)#graph.shape[0],3)
                colors = np.ones((seeds_nodes_graph[i].graph.shape[0],3))
                colors[:,2] = 1 - seeds_nodes_graph[i].value/100
                colors[:,1] = 1 - seeds_nodes_graph[i].value/100
                point_actor = fvtk.point(seeds_nodes_graph[i].graph,colors, point_radius=0.3)
            fvtk.add(r, point_actor)
            #r.add(node_actor)
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
