from __future__ import division

import numpy as np
import vtk
import time
from vtk import*

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


def show_graph_values(FA, streamlines, seeds_nodes_graph, show_final=True):

    #streamlines_actor = actor.line(streamlines)
    time_count = 0
    renderwindow = vtk.vtkRenderWindow()
    renderwindow.SetSize(1000,1000)
    r = window.Renderer()
    #r = vtk.vtkRenderer()
    #r.clear()
    r.background((1, 1, 1))
    #r.SetBackground(1,1,1)
    renderwindow.AddRenderer(r)
    renderwindowInteractor = vtkRenderWindowInteractor()
    #r = vtk.vtkRenderWindowInteractor()
    renderwindowInteractor.SetRenderWindow(renderwindow)
    camera = vtkCamera()
    camera.SetPosition(-50,-50,-50)
    camera.SetFocalPoint(0,0,0)
    #r.SetActiveCamera(camera)
    streamlines_sub = []
    #r.add(streamlines_actor)
    #actor_slicer = actor.slicer(FA, interpolation='nearest')
    #r.add(actor_slicer)
    if not show_final:
        window.show(r)

    colors2 = np.array((0,1,1))
    color3 = np.array((1,1,1))
    colors = np.array((1,0,0))
    colors1 = np.array((0,1,0))
    goal_point = np.array([37,53,59])
    starting_point = np.array([42,55,20])
    goal_point = goal_point[None,:]
    point_actor2 = fvtk.point(goal_point,colors2, point_radius=2)
    point_actor1 = fvtk.point(starting_point[None,:],colors1, point_radius=0.7)
    r.add(point_actor1)
    #r.AddActor(point_actor1)
    r.add(point_actor2)
    #r.AddActor(point_actor2)

    #def time_event(obj, ev):
        #time.sleep(20)
    for i in range(len(seeds_nodes_graph)):
        #print(i)
        #r.clear()
        #node_actor = actor.streamtube([nodes[100]])
        if len(seeds_nodes_graph[i].graph.shape) == 1:
            #colors = np.random.rand(1,3)
            colors = np.array((1,1,1))
            if np.array(seeds_nodes_graph[i].value) > 0:
                colors = np.array((1,1 - seeds_nodes_graph[i].value[0]/100,1 - seeds_nodes_graph[i].value[0]/100))

        else:
            max_value = np.abs(seeds_nodes_graph[i].value.max())
            ss = np.where(seeds_nodes_graph[i].value<0)[0]
            colors = np.ones((seeds_nodes_graph[i].graph.shape[0],3))
            #colors[:,2] = 1 - seeds_nodes_graph[i].value/max_value
            #colors[:,1] = 1 - seeds_nodes_graph[i].value/max_value
            colors[ss,2] = 1
            colors[ss,1] = 1

            sss = np.where(seeds_nodes_graph[i].value>0)[0]
            colors[sss,1] = 0
            colors[sss,2] = 0

            if seeds_nodes_graph[i].value.max()>0:
                #colors = np.zeros((seeds_nodes_graph[i].graph.shape[0],3))
                #colors[:,0] = 1
                streamlines_sub.append(seeds_nodes_graph[i].graph)



        if len(seeds_nodes_graph[i].graph.shape) == 1:
            point_actor = fvtk.point(seeds_nodes_graph[i].graph[None,:],colors, point_radius=0.3)
        else:
            point_actor = fvtk.point(seeds_nodes_graph[i].graph,colors, point_radius=0.3)
        r.add(point_actor)
        #r.AddActor(point_actor)
        #iren = obj
        #iren.GetRenderWindow().Render()
    if not show_final:
        window.show(r)
    peak_slicer = actor.line(streamlines_sub)
    r.add(peak_slicer)
    if show_final:
        window.show(r)
        #renderwindow.Render()
        #renderwindowInteractor.Initialize()
        #renderwindowInteractor.AddObserver('TimerEvent', time_event)
        #timerId = renderwindowInteractor.CreateRepeatingTimer(100)
        #renderwindowInteractor.Start()
"""
colors2 = np.ones((len(streamlines[3]),3))
colors[:,0] = 0
colors[:,1] = 0
point_actor5 = fvtk.point(streamlines[0],colors,point_radius=0.7)
"""

#streamline_actor = fvtk.streamtube(streamlines[3],colors2,linewidth=1.3)




def show_peaks(FA, pam):

    ren = window.Renderer()
    ren.background((1, 1, 1))

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

def show_streamlines_FA(FA, streamlines):

    ren = window.Renderer()
    ren.background((1, 1, 1))
    actor_slicer = actor.slicer(FA, interpolation='nearest')
    #ren.add(actor_slicer)

    #tmp = pam.peak_values
    # tmp[FA<0.2] = 0
    #peak_slicer = actor.peak_slicer(pam.peak_dirs, tmp,
    #                                colors=(1., 0., 0.), lod=False)
    peak_slicer = actor.line(streamlines)
    ren.add(peak_slicer)
    colors2 = np.array((0,1,1))
    color3 = np.array((1,1,1))
    colors = np.array((1,0,0))
    colors1 = np.array((0,1,0))
    goal_point = np.array([37,53,59])
    starting_point = np.array([42,55,20])
    goal_point = goal_point[None,:]
    point_actor2 = fvtk.point(goal_point,colors2, point_radius=2)
    point_actor1 = fvtk.point(starting_point[None,:],colors1, point_radius=0.7)
    ren.add(point_actor1)
    #r.AddActor(point_actor1)
    ren.add(point_actor2)
    #from dipy.reconst.shm import sh_to_sf
    #odf = sh_to_sf(pam.shm_coeff, get_sphere('repulsion724'), 8)
    #odf_slicer = actor.odf_slicer(odf, sphere=get_sphere('repulsion724'))
    window.show(ren)

"""!!!remove graph
"""
def ev_learning(graph, onetrack1, value, reward_positive=1000, reward_negative=-100, positive1=True, alfa=0.2, gamma=0.8):
    """ add positive or negative reward
    """

    if positive1 == True:
        reward1 = reward_positive
    else:
        reward1 = reward_negative

    l1 = len(onetrack1)
    for i in range(l1):
        #!!!replace with tmp = int(onetrack1[l1 - (i +1)])
        if i == 0:
            value[int(onetrack1[l1-(i+1)])] = reward1
        else:
            # Reinforcement learning (TD)
            value[int(onetrack1[l1-(i+1)])] = value[int(onetrack1[l1-(i+1)])] + alfa*(value[int(onetrack1[l1-i])]-value[int(onetrack1[l1-(i+1)])])

    return value


def find_track_point(dirs, track_point, graph, value, direc=True, resolution=1,step_size=1):
    if direc == True:
        track_point_test = track_point + step_size * dirs
    else:
        track_point_test = track_point - step_size * dirs
    if len(graph.shape) == 1:
        norm2 = norm(graph-track_point_test)
    else:
        norm2 = norm(graph-track_point_test,axis=1,ord=2)
    if norm2.min() < resolution:
        index_t = np.argmin(norm2)
        return value[index_t], 1
    else:
        return 0.0, 0

"""
!!! ReinforcementLearningTracking class see the other existing classes
"""
def return_streamline(seed,dirs,track_point,graph,value,with_stand=0,resolution=1,step_size=1):
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
        if norm2.min() < resolution:
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
                    #!!! hard coded parameters!!!!
                    if t0+i > 90 or t0+i == 90:
                        t0 = 90 - i
                    if t1+j > 108 or t1+j == 108:
                        t1 = 108 - j
                    if t2+k > 90 or t2+k == 90:
                        t2 = 90 - k
                    if t0-i < 0:
                        t0 = i
                    if t1 - j<0:
                        t1 = j
                    if t2 - k <0:
                        t2 = k

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
    dir_old = -dirs[t0, t1, t2, 0,:]
    """First direction
    """
    while(FA[t0,t1,t2] != 0 ):
    #    print(track_point)
        #index = 0
        with_stand = -5
        index_inside = 0
        flag_pos_neg1 = 1
        flag_pos_neg2 = 1
        dirc1_2 = 0
        value_single = -500
        t0, t1, t2 = itp(track_point)
        dir_sub = -dirs[t0, t1, t2, 0,:]
        dir_final = -dirs[t0,t1,t2,0,:]
        if dir_sub.all() == False:
        #    print("Im here in picking direction!")
            t0, t1, t2 = check_direction(dirs,t0,t1,t2)
        #    print(t0,t1,t2)
        print("dir_old")
        print(dir_old)
        for i in range(5):
            #t0, t1, t2 = itp(track_point)
            dir_sub = -dirs[t0, t1, t2, i,:]
            print("dir_currnt")
            print(dir_sub)
            if np.dot(dir_old,dir_sub)<0.15:
                    dir_sub = -dir_sub
            #print(i)
            #print(t0,t1,t2)
            #print(dir_sub)
            if dir_sub.all() == True and flag_pos_neg1 == 1:
                value_single_test, out_flag = find_track_point(dir_sub, track_point, graph, value, direc=True, resolution=resolution,step_size=step_size)

                if value_single_test > value_single:
                    index_inside = i
                    value_single = value_single_test
                    dir_final = dir_sub
        if np.dot(dir_old,dir_final)<0.15:
            #print("Im here in break")
            break
        dir_old = dir_final
        track_point = track_point + step_size * dir_final
        if len(graph.shape) == 1:
            norm2 = norm(graph-track_point)
        else:
            norm2 = norm(graph-track_point,axis=1,ord=2)
        if norm2.min() < resolution:
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
            print("Im here in adding value")
            print(value)
            seed_onetrack.track1 = np.append(seed_onetrack.track1, index_t)
            if len(graph.shape) == 1:
                seed_onetrack.nodes1 = np.vstack((seed_onetrack.nodes1, graph))
            else:
                #print(seed_onetrack.nodes1)
                #print("graph")
                #print(graph[int(index_t)])
                seed_onetrack.nodes1 = np.vstack((seed_onetrack.nodes1, graph[int(index_t)]))
        if decision1 == 0:
            stremline = []
        else:
            streamline = np.vstack((streamline,track_point))
        t0, t1, t2 = itp(track_point)
        if t0 > 90 or t0 == 90:
            t0 = 89
        if t1 >108 or t1 == 108:
            t1 = 107
        if t2 > 90 or t2 == 90:
            t2 = 89
        dir_sub = dirs[t0, t1, t2, 0,:]
        if dir_sub.all() == False:
            #print("Im here in picking direction!")
            t0, t1, t2 = check_direction(dirs,t0,t1,t2)
            #print(t0,t1,t2)
    #print("stop at")
    #print(track_point)
    #print("FA value is")
    #print(FA[itp(track_point)])

            #print(t0,t1,t2)
    #print("stop at")
    #print(track_point)
    #print("FA value is")
    #print(FA[itp(track_point)])
    #goal_region1 = [37,53,59]
    goal_region1 = [42,55,20]
    goal_region2 = [37,53,59]
    #goal_region2 = [42,55,20]
    if len(seed_onetrack.nodes1.shape) == 1:
        norm3_track1 = norm(seed_onetrack.nodes1 - goal_region1)
    else:
        norm3_track1 = norm(seed_onetrack.nodes1 - goal_region1,axis=1,ord=2)
    ### !!! fixed parameter
    if norm3_track1.min()<2:
        positive1_test=True
    else:
        positive1_test=False
    """
    if norm3_track2.min() < 10:
        positive2_test=True
    else:
        positive2_test=False
    """
    #print("HERE")
    #print(value)
    value = ev_learning(graph, seed_onetrack.track1, value,positive1=positive1_test)
    return streamline, graph,seed_onetrack, node_onetrack, value, decision1


if __name__ == "__main__":

    """
    vol = np.load('vol_complex_phantom2.npy')
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
    """
#    seeds = utils.seeds_from_mask(seed_mask, density=[2, 2, 2], affine=affine)
    dname = 'C:\\Users\Tingyi\\Dropbox\\Tingyi\\ISMRM_2015_Tracto_challenge_ground_truth_dwi_v2\\ISMRM_2015_Tracto_challenge_ground_truth_dwi_v2\\'

    fbvals = dname +  'NoArtifacts_Relaxation.bvals'
    fbvecs = dname + 'NoArtifacts_Relaxation.bvecs'
    fdwi = dname + 'NoArtifacts_Relaxation.nii.gz'

    from dipy.io.image import load_nifti
    from dipy.io.gradients import read_bvals_bvecs
    from dipy.core.gradients import gradient_table
    from dipy.segment.mask import median_otsu

    bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
    gtab = gradient_table(bvals, bvecs)
    data, affine = load_nifti(fdwi)

    maskdata, mask = median_otsu(data, 3, 1, False, vol_idx=range(1, 20))

    from dipy.reconst.dti import TensorModel

    tensor_model = TensorModel(gtab, fit_method='WLS')
    tensor_fit = tensor_model.fit(data, mask)

    FA = tensor_fit.fa

    from dipy.reconst.shm import CsaOdfModel
    from dipy.data import get_sphere
    from dipy.direction.peaks import peaks_from_model

    sphere = get_sphere('repulsion724')

    csamodel = CsaOdfModel(gtab, 4)
    pam = peaks_from_model(model=csamodel,
                                data=maskdata,
                                sphere=sphere,
                                relative_peak_threshold=.5,
                                min_separation_angle=25,
                                mask=mask,
                                return_odf=False,
                                normalize_peaks=True)


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
    seed = np.array((37,53,59))

    #seed = n[1500]
    """Generate an initial track with it's state values
    """
    graph = seed
    value = [0.0]
    seed_nodes = seed
    #seed_onetrack = Seed(seed, 0)
    #seeds.append(seed_onetrack)
    streamlines_onetrack,graph_onetrack,seed_onetrack, node_onetrack1,value, decision1 = return_streamline(seed,pam.peak_dirs,seed,graph,value,resolution=1, step_size=1)
    streamlines.append(streamlines_onetrack)
    seed_onetrack.nodes = node_onetrack1
    seed_onetrack.index = 0
    #nodes.append(node_onetrack1)
    seeds.append(seed_onetrack)
    one_seed_node_graph = Seed_node_graph(graph_onetrack, value)
    seeds_nodes_graph.append(one_seed_node_graph)
        #graph = graph_onetrack


    grouping_size = 1
    r = window.Renderer()
    r.clear()
    r.background((1, 1, 1))
    colors2 = np.array((0,1,1))
    color3 = np.array((1,1,1))
    colors = np.array((1,0,0))
    colors1 = np.array((0,1,0))
    goal_point = np.array([37,53,59])
    starting_point = np.array([42,55,20])
    goal_point = goal_point[None,:]
    point_actor2 = fvtk.point(goal_point,colors2, point_radius=2)
    point_actor1 = fvtk.point(starting_point[None,:],colors1, point_radius=0.7)
    r.add(point_actor1)
    r.add(point_actor2)


    test = np.array((37,53,59))
#    test = np.array((42,55,20))

#    def Growing_graph(test, n, peak_dirs ,seeds, seed_nodes, seeds_nodes_graph, streamlines):
    for i in range(len(n)):
        seed = n[i]
        #!!! fixed parameter
        if norm(seed - test) < 5:
            if len(seed_nodes.shape) == 1:
                norm2 = norm(seed_nodes-seed)
            if len(seed_nodes.shape) != 1:
                norm2 = norm(seed_nodes-seed,axis=1,ord=2)
            if norm2.min() < grouping_size:
                index_c = np.argmin(norm2)
                streamlines_onetrack, graph_onetrack,seed_onetrack, node_onetrack1,value, decision1 = return_streamline(seed,pam.peak_dirs,seed,seeds_nodes_graph[index_c].graph,seeds_nodes_graph[index_c].value,resolution=1,step_size=1)
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
                streamlines_onetrack, graph_onetrack,seed_onetrack, node_onetrack1, value, decision1 = return_streamline(seed,pam.peak_dirs,seed,graph,value,resolution=1,step_size=1)
                one_seed_node_graph = Seed_node_graph(graph_onetrack, value)
                seeds_nodes_graph.append(one_seed_node_graph)
                seed_onetrack.nodes = node_onetrack1
                seed_onetrack.index = index_c
                seeds.append(seed_onetrack)

            #if decision1 == 1:
            if not streamlines_onetrack == []:
                streamlines.append(streamlines_onetrack)
            #if decision2 == 1:
            #streamlines.append(streamlines_onetrack2)
#    seeds, seeds_nodes_graph, streamlines = Growing_graph(test, n, pam.peak_dirs ,seeds, seed_nodes, seeds_nodes_graph, streamlines)





print('Finished')
