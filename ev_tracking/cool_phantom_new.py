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
from dipy.viz import actor, window
from dipy.viz import fvtk
from dipy.viz.colormap import line_colors



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
        self.nodes1 = []
        self.nodes2 = []

class Seed_node_graph(object):
    def __init__(self, graph, value):
        self.graph = graph
        self.value = value


def build_phantom(fname):

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

    np.save(fname, vol)


def show_graph_values(streamlines, seeds_nodes_graph, show_final=True):

    streamlines_actor = actor.line(streamlines)
    r = window.Renderer()
    r.clear()
    r.background((1, 1, 1))
    r.add(streamlines_actor)
    if not show_final:
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

            r.add(point_actor)

        if not show_final:
            window.show(r)
    if show_final:
        window.show(r)


def show_peaks(FA, pam):

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


def ev_learning(graph, onetrack1, onetrack2, value, reward_positive=100, reward_negative=-100, positive1=True, positive2=True, alfa=0.95, gamma=0.8):
    """ add positive or negative reward
    """

    if positive1 == True:
        reward1 = reward_positive
    else:
        reward1 = reward_negative

    l1 = len(onetrack1)
    for i in range(l1):
        if i == 0:
            value[int(onetrack1[l1-(i+1)])] = reward1
        else:
            # Reinforcement learning (TD)
            value[int(onetrack1[l1-(i+1)])] = value[int(onetrack1[l1-(i+1)])] + alfa*(value[int(onetrack1[l1-i])]-value[int(onetrack1[l1-(i+1)])])

    l2 = len(onetrack2)
    if positive2 == True:
        reward2 = reward_positive
    else:
        reward2 = reward_negative
    for i in range(l2):
        if i == 0:
            value[int(onetrack2[l2-(i+1)])] = reward2
        else:
            # Reinforcement learning (TD)
            value[int(onetrack2[l2-(i+1)])] = value[int(onetrack2[l2-(i+1)])] + alfa*(value[int(onetrack2[l2-i])]-value[int(onetrack2[l2-(i+1)])])

    return value


def find_track_point(dirs, track_point, graph, value, direc=True):
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
    """ returns one streamline, generates graph and does RL
    """
    streamline = seed
    #streamline2 = seed
    node_onetrack = []
    decision1 = 1
    decision2 = 1
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
            value = np.append(value,0.0)
            node_onetrack = seed

    seed_onetrack = Seed(seed, index_c)
    seed_onetrack.track1 = np.append(seed_onetrack.track1, index_c)
    if len(graph.shape) == 1:
        seed_onetrack.nodes1 = graph
    else:
        seed_onetrack.nodes1 = graph[index_c]
    seed_onetrack.track2 = np.append(seed_onetrack.track2, index_c)
    if len(graph.shape) == 1:
        seed_onetrack.nodes2 = graph
    else:
        seed_onetrack.nodes2 = graph[index_c]

    def itp(track_point):
        t0 = int(np.round(track_point[0]))
        t1 = int(np.round(track_point[1]))
        t2 = int(np.round(track_point[2]))
        return t0, t1, t2

    def check_direction(dirs,t0,t1,t2):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    dir_sub = dirs[t0+i, t1+j, t2+k, 0,:]
                    if not dir_sub.all() == False:
                        t0 = t0 + i
                        t1 = t1 + j
                        t2 = t2 + k
                        return t0,t1,t2
                    dir_sub = dirs[t0-i,t1-j,t2-k,0,:]
                    if not dir_sub.all() == False:
                        t0 = t0 - i
                        t1 = t1 - j
                        t2 = t2 - k
                        return t0,t1,t2
        return t0,t1,t2
    t0,t1,t2 = itp(track_point)
    while(FA[t0,t1,t2] != 0 ):
        #index = 0
        with_stand = -5
        index_inside = 0
        flag_pos_neg = 1
        value_single = -500
        #t0, t1, t2 = itp(track_point)
        #dir_sub = dirs[t0, t1, t2, 0,:]
        #if dir_sub.all() == False:
        #    print("Im here in picking direction!")
        #    t0, t1, t2 = check_direction(dirs,t0,t1,t2)
        #    print(t0,t1,t2)
        for i in range(5):
            #t0, t1, t2 = itp(track_point)
            dir_sub = dirs[t0, t1, t2, i,:]
            dir_sub2 = -dirs[t0, t1, t2, i, :]
            if dir_sub.all() == True:
                value_single_test, out_flag = find_track_point(dir_sub, track_point, graph, value, direc=True)

                if value_single_test > value_single:
                    index_inside = i
                    flag_pos_neg = 1
                    value_single = value_single_test
            if dir_sub2.all() == True and i != 0:
                value_single_test, out_flag = find_track_point(dir_sub2, track_point, graph, value, direc=True)

                if value_single_test > value_single:
                    index_inside = i
                    flag_pos_neg = 0
                    value_single = value_single_test

        if flag_pos_neg == 1:
            track_point = track_point + dirs[t0, t1, t2, index_inside,:]
        if flag_pos_neg == 0:
            track_point = track_point - dirs[t0, t1, t2, index_inside,:]
        if len(graph.shape) == 1:
            norm2 = norm(graph-track_point)
        else:
            norm2 = norm(graph-track_point,axis=1,ord=2)
        if norm2.min() < 1:
            index_t = np.argmin(norm2)
            if value[int(index_t)] > with_stand or value[int(index_t)] == with_stand:
                if not np.any(seed_onetrack.track1 == index_t):
                    seed_onetrack.track1 = np.append(seed_onetrack.track1,index_t)
                    if len(graph.shape) == 1:
                        seed_onetrack.nodes1 = np.vstack((seed_onetrack.nodes1, graph))
                    else:
                        seed_onetrack.nodes1 = np.vstack((seed_onetrack.nodes1, graph[int(index_t)]))
            else:
                if not np.any(seed_onetrack.track1 == index_t):
                    seed_onetrack.track1 = np.append(seed_onetrack.track1,index_t)
                    if len(graph.shape) == 1:
                        seed_onetrack.nodes1 = np.vstack((seed_onetrack.nodes1, graph))
                    else:
                        seed_onetrack.nodes1 = np.vstack((seed_onetrack.nodes1, graph[int(index_t)]))
                    decision1 = 0
        else:
            if len(graph.shape) == 1:
                index_t = 1
            else:
                index_t = graph.shape[0]
            graph = np.vstack((graph,track_point))
            value = np.append(value,0.0)
            seed_onetrack.track1 = np.append(seed_onetrack.track1, index_t)
            if len(graph.shape) == 1:
                seed_onetrack.nodes1 = np.vstack((seed_onetrack.nodes1, graph))
            else:
                #print(seed_onetrack.nodes1)
                #print("graph")
                #print(graph[int(index_t)])
                seed_onetrack.nodes1 = np.vstack((seed_onetrack.nodes1, graph[int(index_t)]))

        streamline = np.vstack((streamline,track_point))
        t0, t1, t2 = itp(track_point)
        dir_sub = dirs[t0, t1, t2, 0,:]
        if dir_sub.all() == False:
            #print("Im here in picking direction!")
            t0, t1, t2 = check_direction(dirs,t0,t1,t2)
            #print(t0,t1,t2)
    #print("stop at")
    #print(track_point)
    #print("FA value is")
    #print(FA[itp(track_point)])

    track_point = seed
    t0,t1,t2 = itp(track_point)
    while(FA[t0,t1,t2] != 0):

        with_stand = -5
        index_inside = 0
        flag_pos_neg = 1
        value_single = -500
        t0, t1, t2 = itp(track_point)
        dir_sub = -dirs[t0, t1, t2, 0,:]
        if dir_sub.all() == False:
            t0, t1, t2 = check_direction(dirs,t0,t1,t2)
        for i in range(5):
            #t0, t1, t2 = itp(track_point)
            dir_sub = -dirs[t0, t1, t2, i,:]
            dir_sub2 = dirs[t0, t1, t2, i, :]
            if dir_sub.all() == True:
                value_single_test, out_flag = find_track_point(dir_sub, track_point, graph, value, direc=True)

                if value_single_test > value_single:
                    index_inside = i
                    flag_pos_neg = 1
                    value_single = value_single_test
            if dir_sub2.all() == True and i != 0:
                value_single_test, out_flag = find_track_point(dir_sub2, track_point, graph, value, direc=True)

                if value_single_test > value_single:
                    index_inside = i
                    flag_pos_neg = 0
                    value_single = value_single_test

        if flag_pos_neg == 1:
            track_point = track_point - dirs[t0, t1, t2, index_inside,:]
        if flag_pos_neg == 0:
            track_point = track_point + dirs[t0, t1, t2, index_inside,:]
        #track_point = track_point - dirs[t0, t1, t2, index_inside, :]
        if len(graph.shape) == 1:
            norm2 = norm(graph-track_point)
        else:
            norm2 = norm(graph-track_point,axis=1,ord=2)
        if norm2.min() < 1:
            index_t = np.argmin(norm2)
            if value[int(index_t)] > with_stand or value[int(index_t)] == with_stand:
                if not np.any(seed_onetrack.track2 == index_t):
                    seed_onetrack.track2 = np.append(seed_onetrack.track2,index_t)
                    if len(graph.shape) == 1:
                        seed_onetrack.nodes2 = np.vstack((seed_onetrack.nodes2, graph))
                    else:
                        seed_onetrack.nodes2 = np.vstack((seed_onetrack.nodes2, graph[int(index_t)]))
            else:
                if not np.any(seed_onetrack.track2 == index_t):
                    seed_onetrack.track2 = np.append(seed_onetrack.track2,index_t)
                    if len(graph.shape) == 1:
                        seed_onetrack.nodes2 = np.vstack((seed_onetrack.nodes2, graph))
                    else:
                        seed_onetrack.nodes2 = np.vstack((seed_onetrack.nodes2, graph[int(index_t)]))
                    decision2 = 0
        else:
            if len(graph.shape) == 1:
                index_t = 1
            else:
                index_t = graph.shape[0]
            graph = np.vstack((graph,track_point))
            value = np.append(value,0.0)
            seed_onetrack.track2 = np.append(seed_onetrack.track2, index_t)
            if len(graph.shape) == 1:
                seed_onetrack.nodes2 = np.vstack((seed_onetrack.nodes2, graph))
            else:
                seed_onetrack.nodes2 = np.vstack((seed_onetrack.nodes2, graph[int(index_t)]))

        streamline = np.vstack((streamline,track_point))
        t0, t1, t2 = itp(track_point)
        dir_sub = dirs[t0, t1, t2, 0,:]
        if dir_sub.all() == False:
            #print("Im here in picking direction!")
            t0, t1, t2 = check_direction(dirs,t0,t1,t2)
            #print(t0,t1,t2)
    #print("stop at")
    #print(track_point)
    #print("FA value is")
    #print(FA[itp(track_point)])
    if len(graph.shape) == 1:
        norm3_track1 = norm(graph - [50,79,50])
    else:
        norm3_track1 = norm(graph - [50,79,50],axis=1,ord=2)
    """
    if len(seed_onetrack.nodes2.shape) == 1:
        norm3_track2 = norm(seed_onetrack.nodes2 - [79,50,50])
    else:
        norm3_track2 = norm(seed_onetrack.nodes2 - [79,50,50],axis=1,ord=2)
    """
    if norm3_track1.min() < 10:
        positive1_test=True
        positive2_test=True
    else:
        positive1_test=False
        positive2_test=False
    """
    if norm3_track2.min() < 10:
        positive2_test=True
    else:
        positive2_test=False
    """
    value = ev_learning(graph, seed_onetrack.track1, seed_onetrack.track2, value,positive1=positive1_test,positive2=positive2_test)
    return streamline, graph,seed_onetrack, node_onetrack, value, decision1, decision2


if __name__ == "__main__":

    vol = np.load('vol_reg_cross.npy')
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

    seed_nodes = []
    seeds_nodes_graph = []
    graph = []
    seeds = []
    seeds_index = []
    nodes = []


    #seed = n[1500]
    seed = np.array((20,50,50))
    graph = seed
    value = [0.0]
    seed_nodes = seed
    #seed_onetrack = Seed(seed, 0)
    #seeds.append(seed_onetrack)
    streamlines_onetrack,graph_onetrack,seed_onetrack, node_onetrack1,value, decision1, decision2 = return_streamline(seed,pam.peak_dirs,seed,graph,value)
    streamlines.append(streamlines_onetrack)
    seed_onetrack.nodes = node_onetrack1
    seed_onetrack.index = 0
    #nodes.append(node_onetrack1)
    seeds.append(seed_onetrack)
    one_seed_node_graph = Seed_node_graph(graph_onetrack, value)
    seeds_nodes_graph.append(one_seed_node_graph)
    #graph = graph_onetrack

    for i in range(len(n)):
        seed = n[5]
        test = np.array((20,50,50))
        if norm(seed - test) < 3:
            #print(i)
            #seed = np.array((20,21,50))
            if len(seed_nodes.shape) == 1:
                norm2 = norm(seed_nodes-seed)
            if len(seed_nodes.shape) != 1:
                norm2 = norm(seed_nodes-seed,axis=1,ord=2)
            if norm2.min() < 1:
                index_c = np.argmin(norm2)
                streamlines_onetrack, graph_onetrack,seed_onetrack, node_onetrack1,value, decision1,decision2 = return_streamline(seed,pam.peak_dirs,seed,seeds_nodes_graph[index_c].graph,seeds_nodes_graph[index_c].value)
                #print("streamlines")
                #print(streamlines)
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
                streamlines_onetrack, graph_onetrack,seed_onetrack, node_onetrack1, value, decision1,decision2 = return_streamline(seed,pam.peak_dirs,seed,graph,value)
                one_seed_node_graph = Seed_node_graph(graph_onetrack, value)
                seeds_nodes_graph.append(one_seed_node_graph)
                seed_onetrack.nodes = node_onetrack1
                seed_onetrack.index = index_c
                seeds.append(seed_onetrack)

            #if decision1 == 1:
            streamlines.append(streamlines_onetrack)
            #if decision2 == 1:
            #streamlines.append(streamlines_onetrack2)



print('Finished')
#
