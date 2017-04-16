
from dipy.viz import actor, window

import numpy as np

# from dipy.exp_comparisons_experts import labelize_expert_bundles

from dipy.align.streamlinear import remove_clusters_by_size

from dipy.segment.clustering import QuickBundles

from dipy.align.streamlinear import whole_brain_slr, progressive_slr, BundleMinDistanceStaticMetric

from dipy.tracking.streamline import set_number_of_points

from dipy.tracking.streamline import set_number_of_points

from dipy.tracking.distances import bundles_distances_mdf

from dipy.segment.clustering import QuickBundles

from dipy.segment.metric import MinimumAverageDirectFlipMetric, AveragePointwiseEuclideanMetric


def remove_outliers(streamline_atlas, pruning_thr=15):
    discrete_streamlines = set_number_of_points(streamline_atlas, 20)
    clusters = QuickBundles(threshold=20.,
                            metric=metric).cluster(discrete_streamlines)
    num_centroid = int(round(len(clusters) * 0.4))
    if num_centroid == 0:
        num_centroid = 1
    centroid_atlas = []
    for i in range(num_centroid):
        centroid_atlas += [clusters[i].centroid]
    # from ipdb import set_trace
    # set_trace()
    distance_matrix = bundles_distances_mdf(centroid_atlas,
                                            discrete_streamlines)
    stream_line_min = np.amin(distance_matrix, axis=0)
    s = np.where(stream_line_min < pruning_thr)
    ss = [s[0][i] for i in range(len(s[0]))]

    return [streamline_atlas[i] for i in ss]


def remove_short_streamlines_and_small_clusters(streamlines,
                                                greater_than=50,
                                                less_than=250,
                                                qb_thr=15,
                                                nb_pts=20,
                                                rm_small_clusters=50):

    def check_range(streamline, gt=greater_than, lt=less_than):

        if (len(streamline) > gt) & (len(streamline) < lt):
            return True
        else:
            return False

    streamlines1 = ArraySequence([s for s in streamlines if check_range(s)])

    rstreamlines1 = set_number_of_points(streamlines1, nb_pts)

    qb1 = QuickBundles(threshold=qb_thr, metric=metric)
    rstreamlines1 = [s.astype('f4') for s in rstreamlines1]
    cluster_map1 = qb1.cluster(rstreamlines1)
    clusters1 = remove_clusters_by_size(cluster_map1, rm_small_clusters)

    print('Starting extending')
    reduced = ArraySequence()
    for cluster in clusters1:
        reduced.extend(streamlines1[cluster.indices])

    return reduced

def show_bundle(streamlines, colors, size=(1000, 1000), tubes=False):
    renderer = window.Renderer()
    renderer.background((1, 1, 1))
    if tubes:
        renderer.add(actor.streamtube(streamlines, colors))
    else:
        renderer.add(actor.line(streamlines, colors))
    window.show(renderer, size)


def show_streamlines(streamlines, streamlines2,
                     color_array, color_array2, translate=False, tubes=False):#, record_filename=None):

    renderer = window.Renderer()
    renderer.background((1, 1, 1))
    if not tubes:
        st_actor = actor.line(streamlines, colors=color_array, opacity=1, linewidth=2, spline_subdiv=None, lod=True, lod_points=10 ** 4, lod_points_size=3, lookup_colormap=None)
    else:
        st_actor = actor.streamtube(streamlines, color_array, linewidth=0.5)
    renderer.add(st_actor)

    if not tubes:
        st_actor2 = actor.line(streamlines2, colors=color_array2, opacity=1, linewidth=2,  spline_subdiv=None, lod=True, lod_points=10 ** 4, lod_points_size=3, lookup_colormap=None)
    else:
        st_actor2 = actor.streamtube(streamlines2, color_array2, linewidth=0.5)

    renderer.add(st_actor2)

    if translate:
        st_actor.SetPosition(200, 0, 0)
    else:
        st_actor.SetPosition(0,0,0)
    st_actor2.SetPosition(0,0,0)
    window.show(renderer, title='DIPY', size=(1000, 1000), reset_camera=True, order_transparent=False)

#    if record_filename is not None:
#        window.record(renderer, out_path=record_filename, size=(1000, 1000), magnification=3)


# data_dir = '/Users/tiwanyan/Mount/'
data_dir = '/Users/tiwanyan/Mount/594156/'

dname_atlas = data_dir# + '2013_02_08_Gabriel_Girard/TRK_files/'

dname_full_atlas_streamlines = data_dir + 'prob_int_pft_fodf_tc_ic_20-200.trk'

atlas_dix = {}
atlas_dix['cc_1'] = {'filename': dname_atlas + '594156_CC1_Rostrum.trk'}
atlas_dix['cc_2'] = {'filename': dname_atlas + '594156_CC2_Genu.trk'}
atlas_dix['cc_3'] = {'filename': dname_atlas + '594156_CC3_RostralBody.trk'}
atlas_dix['cc_4'] = {'filename': dname_atlas + '594156_CC4_AnteriorMidbody.trk'}
atlas_dix['cc_5'] = {'filename': dname_atlas + '594156_CC5_PosteriorMidbody.trk'}
atlas_dix['cc_6'] = {'filename': dname_atlas + '594156_CC6_Isthmus.trk'}
atlas_dix['cc_7'] = {'filename': dname_atlas + '594156_CC7_Splenium.trk'}
atlas_dix['cingulum_left'] = {'filename': dname_atlas + '594156_Cingulum_Left.trk'}
atlas_dix['cingulum_right'] = {'filename': dname_atlas + '594156_Cingulum_Right.trk'}
atlas_dix['cst_left'] = {'filename': dname_atlas + '594156_CST_Left.trk'}
atlas_dix['cst_right'] = {'filename': dname_atlas + '594156_CST_Right.trk'}
atlas_dix['fornix'] = {'filename': dname_atlas + '594156_Fornix.trk'}
atlas_dix['ifof_left'] = {'filename': dname_atlas + '594156_IFOF_Left.trk'}
atlas_dix['ifof_right'] = {'filename': dname_atlas + '594156_IFOF_Right.trk'}
atlas_dix['ilf_left'] = {'filename': dname_atlas + '594156_ILF_Left.trk'}
atlas_dix['ilf_right'] = {'filename': dname_atlas + '594156_ILF_Right.trk'}
atlas_dix['mcp'] = {'filename': dname_atlas + '594156_MCP.trk'}
atlas_dix['ort_left'] = {'filename': dname_atlas + '594156_OpticRadiation_Left.trk'}
atlas_dix['ort_right'] = {'filename': dname_atlas + '594156_OpticRadiation_Right.trk'}
atlas_dix['ot'] = {'filename': dname_atlas + '594156_OpticTract.trk'}
atlas_dix['uw_left'] = {'filename': dname_atlas + '594156_UncinateWide_Left.trk'}
atlas_dix['uw_right'] = {'filename': dname_atlas + '594156_UncinateWide_Right.trk'}

"""
atlas_dix['af.left'] = {'filename': dname_atlas + 'bundles_af.left.trk'}
atlas_dix['cc_1'] = {'filename': dname_atlas + 'bundles_cc_1.trk'}
atlas_dix['cc_2'] = {'filename': dname_atlas + 'bundles_cc_2.trk'}
atlas_dix['cc_3'] = {'filename': dname_atlas + 'bundles_cc_3.trk'}
atlas_dix['cc_4'] = {'filename': dname_atlas + 'bundles_cc_4.trk'}
atlas_dix['cc_5'] = {'filename' : dname_atlas + 'bundles_cc_5.trk'}
atlas_dix['cc_6'] = {'filename' : dname_atlas + 'bundles_cc_6.trk'}
atlas_dix['cc_7'] = {'filename' : dname_atlas + 'bundles_cc_7.trk'}
atlas_dix['af.right'] = {'filename' : dname_atlas + 'bundles_af.right.trk'}
atlas_dix['cg.left'] = {'filename' : dname_atlas + 'bundles_cg.left.trk'}
atlas_dix['cg.right'] = {'filename' : dname_atlas + 'bundles_cg.right.trk'}
atlas_dix['cst.left'] = {'filename' : dname_atlas + 'bundles_cst.left.trk'}
atlas_dix['cst.right'] = {'filename' : dname_atlas + 'bundles_cst.right.trk'}
atlas_dix['ifof.left'] = {'filename' : dname_atlas + 'bundles_ifof.left.trk'}
atlas_dix['ifof.right'] = {'filename' : dname_atlas + 'bundles_ifof.right.trk'}
atlas_dix['ilf.left'] = {'filename' : dname_atlas + 'bundles_ilf.left.trk'}
atlas_dix['ilf.right'] = {'filename' : dname_atlas + 'bundles_ilf.right.trk'}
atlas_dix['mdlf.left'] = {'filename' : dname_atlas + 'bundles_mdlf.left.trk'}
atlas_dix['mdlf.right'] = {'filename' : dname_atlas + 'bundles_mdlf.right.trk'}
atlas_dix['slf1.left'] = {'filename' : dname_atlas + 'bundles_slf1.left.trk'}
atlas_dix['slf1.right'] = {'filename' : dname_atlas + 'bundles_slf1.right.trk'}
atlas_dix['slf2.left'] = {'filename' : dname_atlas + 'bundles_slf2.left.trk'}
atlas_dix['slf2.right'] = {'filename' : dname_atlas + 'bundles_slf2.right.trk'}
atlas_dix['slf_3.left'] = {'filename' : dname_atlas + 'bundles_slf_3.left.trk'}
atlas_dix['slf_3.right'] = {'filename' : dname_atlas + 'bundles_slf_3.right.trk'}
atlas_dix['uf.left'] = {'filename' : dname_atlas + 'bundles_uf.left.trk'}
atlas_dix['uf.right'] = {'filename' : dname_atlas + 'bundles_uf.right.trk'}
"""
# streamlines_file = data_dir + '2013_02_08_Gabriel_Girard/streamlines_500K.trk'
streamlines_file = data_dir + 'prob_int_pft_fodf_tc_ic_20-200.trk'


from dipy.io.trackvis import load_trk

from nibabel.streamlines import ArraySequence

full_atlas = ArraySequence()
centroids_atlas = []
centroids_atlas2 = []
color_array = []

#metric = MinimumAverageDirectFlipMetric()
metric = AveragePointwiseEuclideanMetric()

full_atlas2 = []
keys = []
centroids_num = []
total_centroid_num = 0

for key in atlas_dix:
    filename = atlas_dix[key]['filename']

    streamlines, header = load_trk(filename)

    atlas_dix[key]['streamlines'] = streamlines
    discrete_streamlines = set_number_of_points(streamlines, 20)
    discrete_streamlines = remove_outliers(discrete_streamlines, 15)

    clusters = QuickBundles(threshold=15.,
                            metric=metric).cluster(discrete_streamlines)
    atlas_dix[key]['centroids'] = clusters[0].centroid
    lengh_cluster = len(clusters)
    num_centroid = int(round(lengh_cluster * 1))
    if total_centroid_num == 0:
        atlas_dix[key]['low_high'] = [total_centroid_num, num_centroid]
    else:
        atlas_dix[key]['low_high'] = [total_centroid_num - 1, total_centroid_num + num_centroid]
    print(num_centroid)
    if num_centroid == 0:
        num_centroid = 1
    for i in range(num_centroid):
        centroids_atlas += [clusters[i].centroid]

    total_centroid_num = total_centroid_num + num_centroid
    # show_streamlines(streamlines)
    full_atlas.extend(streamlines)
    #full_atlas2.append(discrete_streamlines)
    full_atlas2 += discrete_streamlines
    centroids_num.append(num_centroid)
    keys.append(key)
    # show_streamlines(full_atlas)
    # del streamlines
color_array = np.random.rand(len(centroids_atlas), 3) * 10

streamlines_target, header_target = load_trk(streamlines_file)

print('Large streamlines loaded')
print(len(streamlines_target))
streamlines_target = remove_short_streamlines_and_small_clusters(
    streamlines_target, less_than=300, rm_small_clusters=10)
print(len(streamlines_target))
print('Large streamlines reduced')


def slr(centroids_atlas2, streamlines_target, disp=True, assym=True):
    greater_than = 50
    less_than = 250
    qb_thr = 15
    nb_pts = 20
    rm_small_clusters = 50
    x0 = 'translation'

    #discrete_target = set_number_of_points(streamlines_target, nb_points=20)
    """
    def check_range(streamline, gt=greater_than, lt=less_than):

        if (len(streamline) > gt) & (len(streamline) < lt):
            return True
        else:
            return False

    streamlines1 = [s for s in streamlines_target if check_range(s)]

    rstreamlines1 = streamlines1

    rstreamlines1 = set_number_of_points(rstreamlines1, nb_pts)

    qb1 = QuickBundles(threshold=qb_thr, metric=metric)
    rstreamlines1 = [s.astype('f4') for s in rstreamlines1]
    cluster_map1 = qb1.cluster(rstreamlines1)
    clusters1 = remove_clusters_by_size(cluster_map1, rm_small_clusters)
    qb_centroids = [cluster.centroid for cluster in clusters1]
    """
    bounds = [(-45, 45), (-45, 45), (-45, 45),
              (-30, 30), (-30, 30), (-30, 30),
              (0.6, 1.4), (0.6, 1.4), (0.6, 1.4),
              (-10, 10), (-10, 10), (-10, 10)]

    if assym:
        slr_metric = BundleMinDistanceStaticMetric()
    else:
        slr_metric = None
    slm = progressive_slr(centroids_atlas, streamlines_target,
                          x0='translation', metric=BundleMinDistanceStaticMetric(),
                          bounds=bounds)
    if disp:
        print('Showing SLR result')
        print(len(centroids_atlas2))
        print(len(qb_centroids))

        show_streamlines(centroids_atlas2, qb_centroids,
                         [1, 0., 0], [1, 0.5, 0], tubes=True)
        show_streamlines(centroids_atlas2, slm.transform(qb_centroids),
                         [1, 0., 0], [1, 0.5, 0], tubes=True)
        #from ipdb import set_trace
        #set_trace()

    return slm.transform(streamlines_target)


def dm_pruning(streamlines_target, centroids_atlas2,
               color_array2, threshold, use_slr=True):

    if use_slr:
#        discrete_target = slr(streamlines_target, centroids_atlas2, assym=False)
        discrete_target = slr(streamlines_target, centroids_atlas2, assym=True)
    else:
        discrete_target = set_number_of_points(streamlines_target,
                                               nb_points=20)

    distance_matrix = bundles_distances_mdf(centroids_atlas2, discrete_target)

    index_whole = np.argmin(distance_matrix, axis=0)

    stream_line_min = np.amin(distance_matrix, axis=0)

    index_threshold = np.where(stream_line_min < threshold)

    streamline_final = streamlines_target[index_threshold[0]]

    color_array_target = color_array2[index_whole[index_threshold[0]], :]

#    final_index_whole = index_whole[index_threshold[0]]

    return streamline_final, color_array_target, index_whole, index_threshold, discrete_target


def computing_accuracy(streamline_atlas, streamline_compute):
    distance_matrix = bundles_distances_mdf(streamline_atlas, streamline_compute)

    from ipdb import set_trace
    #set_trace()

    stream_line_min = np.amin(distance_matrix, axis=0)

    index_threshold_accuracy = np.where(stream_line_min < 0.01)

    #set_trace()
    print(len(index_threshold_accuracy[0]))

    return (len(index_threshold_accuracy[0])/np.float(len(streamline_atlas) + len(streamline_compute) - len(index_threshold_accuracy[0])),
            len(index_threshold_accuracy[0])/np.float(len(streamline_atlas)))


def finding_corresponding_index(key, index_whole, index_threshold, streamline_target):

    low = atlas_dix[key]['low_high'][0]

    high = atlas_dix[key]['low_high'][1]
    if low == 0:
        s = np.where(index_whole < high)[0]
    else:
        s = np.where(np.logical_and(index_whole > low, index_whole < high))[0]

    indices = np.where(np.in1d(s, index_threshold[0]))[0]

    # from ipdb import set_trace
    # set_trace()
    ss = s[indices]
    # set_trace()

    streamline_part = streamline_target[ss]

    return ss, indices, streamline_part


def show_detected_bundles(key, index_whole, index_threshold, streamline_target,
                          atlas_part, translate=False):
    low = atlas_dix[key]['low_high'][0]

    high = atlas_dix[key]['low_high'][1]
    if low == 0:
        s = np.where(index_whole < high)[0]
    else:
        s = np.where(np.logical_and(index_whole > low, index_whole < high))[0]

    indices = np.where(np.in1d(s, index_threshold[0]))[0]

    from ipdb import set_trace
    # print(s)
    #set_trace()
    ss = s[indices]
    #set_trace()

    # for discrete
    streamline_part = [streamline_target[i] for i in ss]
    #streamlines_part = [streamlines_target[ss_] for ss_ in ss]
    #streamline_part = streamline_target[ss]
    #set_trace()

    show_streamlines(atlas_part, streamline_part,
                     [0, 0.5, 0], [0, 0, 0.4], translate)


def computing_range_accuracy(key, index_whole, index_threshold,
                             streamline_target, atlas_part):

    ss, indices, streamline_part = finding_corresponding_index(
        key, index_whole, index_threshold, streamline_target)

    streamline_np = np.array(streamline_part, dtype=np.object)

    np.save(key, streamline_np)
    discrete_streamline_part = set_number_of_points(streamline_part, 20)

    jaccard, accuracy = computing_accuracy(atlas_part,
                                           discrete_streamline_part)

    print(key)
    print(jaccard)
    print(accuracy)


def finding_atlas_part(key, centroids_atlas):
    low = atlas_dix[key]['low_high'][0]

    high = atlas_dix[key]['low_high'][1]
    atlas_part = []
    if low == 0:
        s = np.where(np.array(range(len(centroids_atlas))) < high)[0]
    else:
        s = np.where(np.logical_and(np.array(range(len(centroids_atlas))) > low, np.array(range(len(centroids_atlas))) < high))[0]

    for i in range(len(s)):
        atlas_part += [centroids_atlas[s[i]]]

    return atlas_part


if __name__ == '__main__':

    dm_thr = 15
    use_slr = True
    compute_accuracy = False

    print('Streamlines_target is...')
    print(len(streamlines_target))

    atlas_part1 = finding_atlas_part(keys[2], centroids_atlas)
    atlas_part2 = finding_atlas_part(keys[3], centroids_atlas)
    atlas_part3 = finding_atlas_part(keys[4], centroids_atlas)
    atlas_total = atlas_part1 + atlas_part2 + atlas_part3
    #atlas_part1_translate = [atlas_part1[i] + [0, 0, 0] for i in range(len(atlas_part1))]
    #atlas_part2_translate = [atlas_part2[i] + [0, 0, 0] for i in range(len(atlas_part2))]
    #atlas_part3_translate = [atlas_part3[i] + [5, 0, 0] for i in range(len(atlas_part3))]
    #atlas_whole_translate = [centroids_atlas[i] + [5, 0, 0] for i in range(len(centroids_atlas))]
    #atlas_total_translate = atlas_part1_translate + atlas_part2_translate

    res = slr(centroids_atlas, atlas_total, disp=False, assym=True)
    res2 = slr(centroids_atlas, atlas_part2, disp=False, assym=True)
    #res = dm_pruning(atlas_part1, centroids_atlas,
    #                 color_array, dm_thr, use_slr)
    #streamline_final, color_array_target, index_whole, index_threshold, discrete_streamlines_target = res

    #discrete_streamlines_target = set_number_of_points(streamlines_target, 20)

    # from ipdb import set_trace
    # set_trace()
"""
    for i in range(27):
        if keys[i] == 'ifof.left':
            show_detected_bundles(keys[i], index_whole, index_threshold,
                                  discrete_streamlines_target, full_atlas2[i],
                                  True)

        if compute_accuracy:
            computing_range_accuracy(keys[i], index_whole, index_threshold,
                                     streamlines_target,
                                     atlas_part=full_atlas2[i])
"""
