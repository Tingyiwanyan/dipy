""" Compare recognition with expert neuroanatomists from Bordeaux
"""
import os
import numpy as np
from dipy.workflows.align import whole_brain_slr_flow
from dipy.workflows.segment import recognize_bundles_flow
from dipy.workflows.segment import assign_bundle_labels_flow
from glob import glob
from time import time
from ipdb import set_trace
import nibabel as nib
#from exp_ismrm2015_bundles import show_bundles, show_grid
from dipy.tracking.distances import bundles_distances_mdf
from dipy.tracking.streamline import set_number_of_points
import pandas as pd
from dipy.io.pickles import save_pickle
from dipy.workflows.base import IntrospectiveArgumentParser


def labelize_expert_bundles(full, bundle, bundle_labels_npy):

#    full_trk_file = nib.streamlines.load(full_trk)
#   full = full_trk_file.streamlines

#    bundle_trk_file = nib.streamlines.load(bundle_trk)
#    bundle = bundle_trk_file.streamlines

#    print(en(full))
#    print(len(bundle))

    bundle_labels = []

    for (j, b) in enumerate(bundle):
        #        print(100 * j/float(len(bundle)))
        for (i, s) in enumerate(full):

            if s.shape[0] != b.shape[0]:
                continue
            first_pt = np.linalg.norm(b[0] - s[0])
            if first_pt > 0.01:
                continue
            last_pt = np.linalg.norm(b[-1] - s[-1])
            mid_pt = np.linalg.norm(
                            b[b.shape[0] / 2] - s[s.shape[0] / 2])
            tmp = (first_pt + last_pt + mid_pt) / 3.
            if tmp <= 0.001:
                bundle_labels.append(i)

    bundle_labels = list(set(bundle_labels))
#    del full
#    del full_trk_file

    return bundle_labels_npy, np.array(bundle_labels)
    

def bundle_adjacency(dtracks0, dtracks1, threshold):
    # d01 = distance_matrix(MinimumAverageDirectFlipMetric(),
    #                       dtracks0, dtracks1)
    d01 = bundles_distances_mdf(dtracks0, dtracks1)

    pair12 = []
    solo1 = []

    for i in range(len(dtracks0)):
        if np.min(d01[i, :]) < threshold:
            j = np.argmin(d01[i, :])
            pair12.append((i, j))
        else:
            solo1.append(dtracks0[i])

    pair12 = np.array(pair12)
    pair21 = []

    solo2 = []
    for i in range(len(dtracks1)):
        if np.min(d01[:, i]) < threshold:
            j = np.argmin(d01[:, i])
            pair21.append((i, j))
        else:
            solo2.append(dtracks1[i])

    pair21 = np.array(pair21)
    A = len(pair12) / np.float(len(dtracks0))
    B = len(pair21) / np.float(len(dtracks1))
    res = 0.5 * (A + B)
    return res


def ba_analysis(recognized_trk, expert_trk, threshold=2.):
    recognized_trk_file = nib.streamlines.load(recognized_trk)
    recognized_bundle = recognized_trk_file.streamlines
    recognized_bundle = set_number_of_points(recognized_bundle, 20)

    expert_trk_file = nib.streamlines.load(expert_trk)
    expert_bundle = expert_trk_file.streamlines
    expert_bundle = set_number_of_points(expert_bundle, 20)

    return bundle_adjacency(recognized_bundle, expert_bundle, threshold)


def spatial_extent(recognized_trk, expert_trk):

    recognized_trk_file = nib.streamlines.load(recognized_trk)
    recognized_bundle = recognized_trk_file.streamlines
    recognized_bundle = set_number_of_points(recognized_bundle, 20)

    expert_trk_file = nib.streamlines.load(expert_trk)
    expert_bundle = expert_trk_file.streamlines
    expert_bundle = set_number_of_points(expert_bundle, 20)

    vol_size = (256, 256, 256)
    vol = np.zeros(vol_size)

    rpts = recognized_bundle._data
    epts = expert_bundle._data

    A = np.zeros((2, 3))
    A[0] = rpts.min(axis=0)
    A[1] = epts.min(axis=0)

    shift = A.min(axis=0)

    rpts = rpts - shift + 1
    epts = epts - shift + 1

    rptsi = np.round(rpts).astype(np.int)
    eptsi = np.round(epts).astype(np.int)

    for index in rptsi:
        i, j, k = index
        vol[i, j, k] = 1

    vol2 = np.zeros(vol_size)
    for index in eptsi:
        i, j, k = index
        vol2[i, j, k] = 1

    vol_and = np.logical_and(vol, vol2)
    # overlap = np.sum(vol_and) / float(np.sum(vol2))

    vol_or = np.logical_or(vol, vol2)
    jaccard = np.sum(vol_and) / float(np.sum(vol_or))

    sensitivity = np.sum(vol_and) / np.sum(vol2)

    rec_vox_count = np.sum(vol)
    exp_vox_count = np.sum(vol2)

    # if show:
    #     viz_vol(vol_and)
    return rec_vox_count, exp_vox_count, sensitivity, jaccard


def binary_classification(recognized_npy, expert_npy, expert_wb_trk):

    # https://en.wikipedia.org/wiki/sensitivity_and_specificity
    rec_labels = np.load(recognized_npy)
    exp_labels = np.load(expert_npy)
    full_trk_file = nib.streamlines.load(expert_wb_trk)
    expert_wb_size = len(full_trk_file.streamlines)
    del full_trk_file

    tp = len(np.intersect1d(rec_labels, exp_labels))
    fp = len(np.setdiff1d(rec_labels, exp_labels))
    fn = len(np.setdiff1d(exp_labels, rec_labels))
    tn = len(np.setdiff1d(range(expert_wb_size),
                          np.union1d(rec_labels, exp_labels)))
    if tp == 0:
        sensitivity = np.nan
        specificity = np.nan
        precision = np.nan
        fdr = np.nan
        f1 = np.nan
        accuracy = np.nan
        jaccard = np.nan
    else:
        sensitivity = tp / float(tp + fn)
        specificity = tn / float(tn + fp)
        precision = tp / float(tp + fp)
        fdr = fp / float(tp + fp)
        f1 = 2 * tp / float(2 * tp + fp + fn)
        accuracy = (tp + tn) / float(tp + fp + fn + tn)
        jaccard = tp / float(len(np.union1d(rec_labels, exp_labels)))

    res = {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'jaccard': jaccard,
           'sensitivity': sensitivity, 'specificity': specificity,
           'precision': precision, 'fdr': fdr, 'f1': f1,
           'accuracy': accuracy,
           'rec_size': len(rec_labels),
           'exp_size': len(exp_labels)}
    return res


def analyze(recognized_trk, expert_trk, model_trk):
    """ This is the function that calculates the success of classification
    """

    recognized_trk_file = nib.streamlines.load(recognized_trk)
    recognized_bundle = recognized_trk_file.streamlines

    expert_trk_file = nib.streamlines.load(expert_trk)
    expert_bundle = expert_trk_file.streamlines

    model_trk_file = nib.streamlines.load(model_trk)
    model_bundle = model_trk_file.streamlines

    rec_len = len(recognized_bundle)
    exp_len = len(expert_bundle)

    DM = np.inf * np.ones((rec_len, exp_len))

    for i, rec_b in enumerate(recognized_bundle):
        for j, exp_b in enumerate(expert_bundle):
            first_pt = np.linalg.norm(rec_b[0] - exp_b[0])
            last_pt = np.linalg.norm(rec_b[-1] - exp_b[-1])
            mid_pt = np.linalg.norm(
                rec_b[rec_b.shape[0] / 2] - exp_b[exp_b.shape[0] / 2])
            tmp = (first_pt + last_pt + mid_pt) / 3.
            if tmp <= 0.001:
                DM[i, j] = tmp

    print(np.sum(DM < np.inf))
    same_streamlines_idx = np.where(DM < np.inf)
    print(len(expert_bundle))
    print(len(recognized_bundle))

    # I = np.argmin(DM, axis=0)
    # J = np.argmin(DM, axis=1)
    # show_bundles(recognized_bundle[list(same_streamlines_idx[0])],
    #              expert_bundle[list(same_streamlines_idx[1])])
    # show_grid([recognized_bundle, expert_bundle, model_bundle],
    #           list_of_captions=['Recognized', 'Expert', 'Model'])
    # set_trace()
    # jaccard = len(I) + len(J)
    ratio_exp_bundle = len(same_streamlines_idx[0]) / float(len(expert_bundle))
    ratio_rec_bundle = len(same_streamlines_idx[0]) / \
        float(len(recognized_bundle))

    return ratio_exp_bundle, ratio_rec_bundle


def summarize_xlsx(bundle_type='UNC_L', slr=True, pruning_thr=8.):
    dname_results = '/home/eleftherios/Data/Hackethon_bdx/' + \
        'automatic_extraction_test_' + bundle_type + '_slr_' + str(slr) + \
        '_prun_thr_' + str(pruning_thr) + '/'

    fxlsx = dname_results + 'table.xlsx'

    sheets = ['Sensitivity', 'Specificity', 'Accuracy', 'Precision',
              'F1', 'FDR', 'Jaccard', 'Bundle_Adjacency', 'Bundle_Adjacency_4',
              'Rec_size', 'Exp_size']
    # np.set_printoptions(3, suppress=True)
    pd.set_option('display.height', 1000)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.set_option('display.precision', 3)

    df_overall = pd.DataFrame(columns=sheets,
                              index=['mean'])
    for sheet in sheets:
        df = pd.read_excel(fxlsx, sheetname=sheet)
        # print(sheet)
        # print(df.loc['mean'].mean())
        df_overall[sheet]['mean'] = df.loc['mean'].mean()

    print(df_overall)


def cross_validation(data_folder, out_folder,
                     subdir='automatic_extraction_test_',
                     bundle_type='UNC_L', verbose=False,
                     slr=True, metric=None,
                     matrix='small', pruning_thr=8., debug=False,
                     choose_target=10,
                     choose_model=10):

    # model_tag = 't0175'  # 't0502'  # 't0337'
    # bundle_type = 'UNC_L'  # 'IFOF_R'
    dname_model_bundles = data_folder + \
        '/bordeaux_tracts_and_stems/'

    dname_whole_tracks = data_folder + \
        'bordeaux_whole_brain_DTI_corrected/whole_brain_trks_60sj/'

    dname_results = out_folder + \
        subdir + bundle_type + '_slr_' + str(slr) + \
        '_prun_thr_' + str(pruning_thr) + '/'

    if not os.path.exists(dname_results):
        os.mkdir(dname_results)

    fxlsx = dname_results + 'table.xlsx'

    labelize = False
    class_results = {}
    class_results[bundle_type] = {}

    all_tag_names = [os.path.basename(m)
                     for m in glob(dname_model_bundles + '*')]

    rng = np.random.RandomState(42)

    all_tag_names_choice = list(rng.choice(all_tag_names, choose_target,
                                replace=False))
    # all_tag_names_choice = ['t0501', 't0381']
    # all_tag_names_choice = ['t0381', 't0480', 't0412']
    # all_tag_names_choice = ['t0175', 't0377', 't0340', 't0359', 't0362', 't0337', 't0363', 't0353', 't0345', 't0371']

    all_tag_names_mean_std = all_tag_names_choice + ['mean', 'std']

    # print('All tag names')
    # print(all_tag_names)

    # model_tag_names = ['t0336', 't0337', 't0340', 't0345', 't0364']
    rng2 = np.random.RandomState(43)
    model_tag_names = list(
        rng2.choice(all_tag_names, choose_model, replace=False))
    # model_tag_names = all_tag_names_choice
    # model_tag_names = ['t0126'] + ['t0336']
    # model_tag_names = ['t0395']  # ['t0381', 't0395']
    # model_tag_names = ['t0381'] + ['t0501']

    df_sensitivity = pd.DataFrame(columns=model_tag_names,
                                  index=all_tag_names_mean_std)

    df_specificity = pd.DataFrame(columns=model_tag_names,
                                  index=all_tag_names_mean_std)

    df_accuracy = pd.DataFrame(columns=model_tag_names,
                               index=all_tag_names_mean_std)

    df_precision = pd.DataFrame(columns=model_tag_names,
                                index=all_tag_names_mean_std)

    df_f1 = pd.DataFrame(columns=model_tag_names,
                         index=all_tag_names_mean_std)

    df_fdr = pd.DataFrame(columns=model_tag_names,
                          index=all_tag_names_mean_std)

    df_jaccard = pd.DataFrame(columns=model_tag_names,
                              index=all_tag_names_mean_std)

    df_ba = pd.DataFrame(columns=model_tag_names,
                         index=all_tag_names_mean_std)

    df_ba_4 = pd.DataFrame(columns=model_tag_names,
                           index=all_tag_names_mean_std)

    df_rec_size = pd.DataFrame(columns=model_tag_names,
                               index=all_tag_names_mean_std)
    df_exp_size = pd.DataFrame(columns=model_tag_names,
                               index=all_tag_names_mean_std)

    df_se_rec_vox = pd.DataFrame(columns=model_tag_names,
                                 index=all_tag_names_mean_std)

    df_se_exp_vox = pd.DataFrame(columns=model_tag_names,
                                 index=all_tag_names_mean_std)

    df_se_sensitivity = pd.DataFrame(columns=model_tag_names,
                                     index=all_tag_names_mean_std)

    df_se_jaccard = pd.DataFrame(columns=model_tag_names,
                                 index=all_tag_names_mean_std)

    for (i, model_tag) in enumerate(model_tag_names):

        class_results[bundle_type][model_tag] = {}

        model_bundle_trk = dname_model_bundles + \
            model_tag + '/tracts/' + \
            bundle_type + '/' + model_tag + \
            '_' + bundle_type + '_GP.trk'

        # model_bundle = read_trk(model_bundle_trk)

        if not os.path.exists(dname_results):
            os.mkdir(dname_results)

        ending_name = '_dti_mean02_fact-45_splined.trk'

        wb_trk1 = dname_whole_tracks + model_tag + ending_name
        static_basename = os.path.splitext(
            os.path.basename(wb_trk1))[0]

        for (j, current_tag) in enumerate(all_tag_names_choice):

            wb_trk2 = dname_whole_tracks + current_tag + ending_name
            moving_basename = os.path.splitext(os.path.basename(wb_trk2))[0]
            moving_tag = moving_basename.split('_')[0]

            # if moving_tag != 't0341':
            #     continue

            print('>>> Whole brain SLR')
            print(' Moving')
            print(wb_trk2)
            print(' Static (Model)')
            print(wb_trk1)

            whole_brain_slr_flow(wb_trk2, wb_trk1, out_dir=dname_results,
                                 verbose=verbose)
            # moved_bundle_file = os.path.join(dname_results,
            #                                 moving_basename + '_moved.trk')
            moved_streamlines_file = os.path.join(
                dname_results, moving_basename + '__to__' + static_basename + '.trk')

            recognize_bundles_flow(moved_streamlines_file,
                                   model_bundle_trk,
                                   out_dir=dname_results,
                                   clust_thr=15.,
                                   reduction_thr=20.,
                                   reduction_distance='mdf',
                                   model_clust_thr=5.,
                                   pruning_thr=pruning_thr,
                                   pruning_distance='mdf',
                                   slr=slr,
                                   slr_metric=metric,
                                   slr_transform='rigid',
                                   slr_progressive=True,
                                   slr_matrix=matrix,
                                   verbose=verbose,
                                   debug=debug)

            base_sf = os.path.splitext(
                os.path.basename(moved_streamlines_file))[0]
            base_mb = os.path.splitext(os.path.basename(model_bundle_trk))[0]

            recognized_bundle_trk = os.path.join(
                dname_results,
                base_mb + '_of_' + base_sf + '.trk')

            recognized_bundle_labels_npy = os.path.join(
                dname_results,
                base_mb + '_of_' + base_sf + '_labels.npy')

            assign_bundle_labels_flow(wb_trk2, recognized_bundle_labels_npy)

            labels_base = os.path.splitext(
                os.path.basename(recognized_bundle_labels_npy))[0].split('_labels')[0]

            recognized_bundle_final_trk = os.path.join(
                dname_results,
                labels_base + '_of_' + os.path.basename(wb_trk2))

            # recognized_bundle_final_trk = os.path.join(
            #     dname_results,
            #
            #     base_mb + '_of_' + base_sf + '_of_' + base_sf.split('_moved')[0] + '.trk')

            if not os.path.exists(recognized_bundle_trk):
                raise ValueError("Recognized trk does not exist")

            expert_bundle_trk = dname_model_bundles + \
                moving_tag + '/tracts/' + \
                bundle_type + '/' + moving_basename.split('_')[0] + \
                '_' + bundle_type + '_GP.trk'

            expert_bundle_npy = dname_model_bundles + \
                moving_tag + '/tracts/' + \
                bundle_type + '/' + moving_basename.split('_')[0] + \
                '_' + bundle_type + '_GP.npy'

            if labelize:
                print('Labelizing...')
                labelize_expert_bundles(wb_trk2, expert_bundle_trk,
                                        expert_bundle_npy)
                print(expert_bundle_npy)
                continue

            if not os.path.exists(expert_bundle_trk):
                raise ValueError("Recognized trk does not exist")

            print('Recognized bundle final trk')
            print(recognized_bundle_final_trk)

            print('Expert bundle trk')
            print(expert_bundle_trk)

            bc = binary_classification(recognized_bundle_labels_npy,
                                       expert_bundle_npy,
                                       wb_trk2)
            ba = ba_analysis(recognized_bundle_final_trk,
                             expert_bundle_trk, 2.)

            ba_4 = ba_analysis(recognized_bundle_final_trk,
                               expert_bundle_trk, 4.)
            se = spatial_extent(recognized_bundle_final_trk,
                                expert_bundle_trk)
            # if np.isnan(bc['sensitivity']):
            #    1/0

            df_sensitivity[model_tag][current_tag] = bc['sensitivity']
            df_specificity[model_tag][current_tag] = bc['specificity']
            df_accuracy[model_tag][current_tag] = bc['accuracy']
            df_precision[model_tag][current_tag] = bc['precision']
            df_f1[model_tag][current_tag] = bc['f1']
            df_fdr[model_tag][current_tag] = bc['fdr']
            df_jaccard[model_tag][current_tag] = bc['jaccard']
            df_ba[model_tag][current_tag] = ba
            df_ba_4[model_tag][current_tag] = ba_4
            df_rec_size[model_tag][current_tag] = bc['rec_size']
            df_exp_size[model_tag][current_tag] = bc['exp_size']
            df_se_rec_vox[model_tag][current_tag] = se[0]
            df_se_exp_vox[model_tag][current_tag] = se[1]
            df_se_sensitivity[model_tag][current_tag] = se[2]
            df_se_jaccard[model_tag][current_tag] = se[3]

    writer = pd.ExcelWriter(fxlsx)

    std = df_sensitivity.std()
    df_sensitivity.loc['mean'] = df_sensitivity.mean()
    df_sensitivity.loc['std'] = std
    df_sensitivity.to_excel(excel_writer=writer,
                            sheet_name='Sensitivity')

    std = df_specificity.std()
    df_specificity.loc['mean'] = df_specificity.mean()
    df_specificity.loc['std'] = std
    df_specificity.to_excel(excel_writer=writer,
                            sheet_name='Specificity')

    std = df_accuracy.std()
    df_accuracy.loc['mean'] = df_accuracy.mean()
    df_accuracy.loc['std'] = std
    df_accuracy.to_excel(excel_writer=writer,
                         sheet_name='Accuracy')

    std = df_precision.std()
    df_precision.loc['mean'] = df_precision.mean()
    df_precision.loc['std'] = std
    df_precision.to_excel(excel_writer=writer,
                          sheet_name='Precision')

    std = df_f1.std()
    df_f1.loc['mean'] = df_f1.mean()
    df_f1.loc['std'] = std
    df_f1.to_excel(excel_writer=writer,
                   sheet_name='F1')

    std = df_fdr.std()
    df_fdr.loc['mean'] = df_fdr.mean()
    df_fdr.loc['std'] = std
    df_fdr.to_excel(excel_writer=writer,
                    sheet_name='FDR')

    std = df_jaccard.std()
    df_jaccard.loc['mean'] = df_jaccard.mean()
    df_jaccard.loc['std'] = std
    df_jaccard.to_excel(excel_writer=writer,
                        sheet_name='Jaccard')

    std = df_ba.std()
    df_ba.loc['mean'] = df_ba.mean()
    df_ba.loc['std'] = std
    df_ba.to_excel(excel_writer=writer,
                   sheet_name='Bundle_Adjacency')

    std = df_ba_4.std()
    df_ba_4.loc['mean'] = df_ba_4.mean()
    df_ba_4.loc['std'] = std
    df_ba_4.to_excel(excel_writer=writer,
                     sheet_name='Bundle_Adjacency_4')

    std = df_rec_size.std()
    df_rec_size.loc['mean'] = df_rec_size.mean()
    df_rec_size.loc['std'] = std
    df_rec_size.to_excel(excel_writer=writer,
                         sheet_name='Rec_size')

    std = df_exp_size.std()
    df_exp_size.loc['mean'] = df_exp_size.mean()
    df_exp_size.loc['std'] = std
    df_exp_size.to_excel(excel_writer=writer,
                         sheet_name='Exp_size')

    std = df_se_rec_vox.std()
    df_se_rec_vox.loc['mean'] = df_se_rec_vox.mean()
    df_se_rec_vox.loc['std'] = std
    df_se_rec_vox.to_excel(excel_writer=writer,
                           sheet_name='se_rec')

    std = df_se_exp_vox.std()
    df_se_exp_vox.loc['mean'] = df_se_exp_vox.mean()
    df_se_exp_vox.loc['std'] = std
    df_se_exp_vox.to_excel(excel_writer=writer,
                           sheet_name='se_exp')

    std = df_se_sensitivity.std()
    df_se_sensitivity.loc['mean'] = df_se_sensitivity.mean()
    df_se_sensitivity.loc['std'] = std
    df_se_sensitivity.to_excel(excel_writer=writer,
                               sheet_name='se_sensitivity')

    std = df_se_jaccard.std()
    df_se_jaccard.loc['mean'] = df_se_jaccard.mean()
    df_se_jaccard.loc['std'] = std
    df_se_jaccard.to_excel(excel_writer=writer,
                           sheet_name='se_jaccard')

    writer.save()
    print('Results saved in {}'.format(fxlsx))


def main():

    cv = True
    summarize = False
    verbose = True
    debug = True
    choose_model = 10
    choose_target = 10
    slr = True
    metric = 'asymmetric'
    matrix = 'huge'
    data_folder = '/home/eleftherios/Data/Hackethon_bdx/'
    out_folder = '/home/eleftherios/Data/Hackethon_bdx/'
    subdir = 'results_same_test_' + str(slr) + '_' + metric + '_' + \
        matrix + '_' + str(choose_model) + '_' + str(choose_target) + '_'

    for bundle_type in ['IFOF_L', 'IFOF_R', 'UNC_L', 'UNC_R']:
        print('')
        print('===========================')
        print('Bundle type {}'.format(bundle_type))
        print('===========================')

        for pruning_thr in [8., 6.]:
            print('')
            print('Pruning {}'.format(pruning_thr))
            print('===========================')

            if cv:
                cross_validation(data_folder, out_folder, subdir,
                                 bundle_type=bundle_type, verbose=verbose,
                                 slr=slr, metric=metric, matrix=matrix,
                                 pruning_thr=pruning_thr,
                                 debug=debug,
                                 choose_model=choose_model,
                                 choose_target=choose_target)
            if summarize:
                summarize_xlsx(bundle_type=bundle_type,
                               slr=True, pruning_thr=pruning_thr)


def cross_validation_flow(data_folder, out_folder, choose_model=3, choose_target=2, slr=1, metric='symmetric', matrix='small', bundle_type='IFOF_L', pruning_thr=10.):
    """ Cross validation flow

    Parameters
    ----------
    data_folder : str
        Path to data folder
    out_folder : str
        Path to output folder
    choose_model : int
        Default 3
    choose_target : int
        Default 2
    slr : int
        Default 1
    metric : string
        Default 'symmetric'
    matrix : string
        Default 'small'.
    bundle_type : string
        Default 'IFOF_L'
    pruning_thr : float
        Default 10.0

    """

    verbose = True
    debug = True
    if slr == 1:
        slr = True
    else:
        slr = False
    subdir = 'results_' + str(slr) + '_' + metric + '_' + \
        matrix + '_' + str(choose_model) + '_' + str(choose_target) + '_'

    cross_validation(data_folder, out_folder, subdir,
                     bundle_type=bundle_type, verbose=verbose,
                     slr=slr, metric=metric, matrix=matrix,
                     pruning_thr=pruning_thr,
                     debug=debug,
                     choose_model=choose_model,
                     choose_target=choose_target)



def summarize_xlsx_mammoth():

    base_dir = '/home/eleftherios/Data/Mammoth/mnt/' + \
        'parallel_scratch_mp2_wipe_on_august_2016/descotea/' + \
        'coteharn/recobundles/results/'

    choose_model = 10
    choose_target = 10
    slr = False
    metric = 'symmetric'
    matrix = 'small'
    bundle_type = 'UNC_L'
    pruning_thr = 10.

    all_dfs = []

    for bundle_type in ['UNC_L', 'UNC_R', 'IFOF_L', 'IFOF_R']:
        for pruning_thr in [6., 8., 10.]:
            # for metric in ['symmetric', 'asymmetric']:

                print('Metric {} pruning_thr {}'.format(metric, pruning_thr))
                dname_results = 'results_' + str(slr) + '_' + metric +\
                    '_' + matrix + '_' + \
                    str(choose_model) + '_' + str(choose_target) +\
                    '_' + bundle_type + \
                    '_slr_' + str(slr) + '_prun_thr_' + str(pruning_thr) + '/'

                fxlsx = base_dir + dname_results + 'table.xlsx'
                # set_trace()

                sheets = ['Sensitivity', 'Specificity', 'Accuracy',
                          'Precision',
                          'F1', 'FDR', 'Jaccard', 'Bundle_Adjacency',
                          'Bundle_Adjacency_4',
                          'Rec_size', 'Exp_size']

                # np.set_printoptions(3, suppress=True)
                pd.set_option('display.height', 1000)
                pd.set_option('display.max_rows', 500)
                pd.set_option('display.max_columns', 500)
                pd.set_option('display.width', 1000)
                pd.set_option('display.precision', 3)

                index_name = bundle_type + '_' + \
                    str(slr) + '_' + matrix + '_' + \
                    metric + '_' + str(pruning_thr)

                df_overall = pd.DataFrame(columns=sheets,
                                          index=[index_name])
                for sheet in sheets:
                    df = pd.read_excel(fxlsx, sheetname=sheet)
                    # print(sheet)
                    # print(df.loc['mean'].mean())
                    df_overall[sheet][index_name] = df.loc['mean'].mean()
                    # df_overall[sheet]['std'] = df.loc['std'].mean()
                    # df_overall[sheet]['median'] = df[:-2].median().median()

                # print(dname_results)
                # print(df_overall)
                all_dfs.append(df_overall)

                # 1/0

    sum_df = pd.concat(all_dfs)
    print(sum_df)


def overall(fxlsx, bundle_type):

    # sheets = ['Sensitivity', 'Specificity', 'Accuracy', 'Precision',
    #           'F1', 'FDR', 'Jaccard', 'Bundle_Adjacency', # 'Bundle_Adjacency_4',
    #           'Rec_size', 'Exp_size',
    #           'se_rec', 'se_exp',
    #           'se_sensitivity', 'se_jaccard']

    sheets = ['Sensitivity', 'Specificity', 'Accuracy', 'Precision',
              'Jaccard', 'Bundle_Adjacency',
              'Rec_size', 'Exp_size',
              'se_rec', 'se_exp',
              'se_sensitivity', 'se_jaccard']

    # np.set_printoptions(3, suppress=True)
    pd.set_option('display.height', 1000)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.set_option('display.precision', 3)

    df_overall = pd.DataFrame(columns=sheets,
                              index=[bundle_type + '_mean'])
    for sheet in sheets:
        df = pd.read_excel(fxlsx, sheetname=sheet)
        # print(sheet)
        # print(df.loc['mean'].mean())
        tmp = str(np.round(df.loc['mean'].mean(), 2)) + ' ' + '+' + ' ' + str(np.round(df.loc['std'].mean(), 2))
        df_overall[sheet][bundle_type + '_mean'] = tmp
        # if sheet in ['Rec_size', 'Exp_size', 'se_rec', 'se_exp']:
        #     df_overall[sheet] = df_overall[sheet].map('{:,.0f}'.format)
        # else:
        #     df_overall[sheet] = df_overall[sheet].map('{:,.3f}'.format)

    return df_overall


def final_table():

    f_ifof_l = '../results/results_se/table_IFOF_L_prun_8.xlsx'
    f_ifof_r = '../results/results_se/table_IFOF_R_prun_8.xlsx'
    f_unc_l = '../results/results_se/table_UNC_L_prun_8.xlsx'
    f_unc_r = '../results/results_se/table_UNC_R_prun_8.xlsx'

    d1 = overall(f_ifof_l, 'IFOF_L')
    d2 = overall(f_ifof_r, 'IFOF_R')
    d3 = overall(f_unc_l, 'UNC_L')
    d4 = overall(f_unc_r, 'UNC_R')

    final = pd.concat([d1, d2, d3, d4])
    print(final)
    1/0


def find_argmax_and_show():

    bundle_type = 'IFOF_R'
    f_ifof_l = '../results/results_se/table_' + bundle_type + '_prun_8.xlsx'
    df = pd.read_excel(f_ifof_l, sheetname='Accuracy')
    #1/0
    df_imax = df.idxmax()
    print(df_imax)
    # set_trace()
    # model_tag = 't0365'  # 't0377'
    # rec_tag = 't0353'  # 't0375'

    def get_files(bundle_type, model_tag, rec_tag):
        slr = True
        metric = 'symmetric'
        matrix = 'small'
        choose_model = 20
        choose_target = 60
        pruning_thr = 8.0

        #set_trace()

        subdir = 'results_' + str(slr) + '_' + metric + '_' + \
            matrix + '_' + str(choose_model) + '_' + str(choose_target) + '_'

        ending_name = '_dti_mean02_fact-45_splined.trk'

        out_folder = '/home/eleftherios/Data/Mammoth/mnt/parallel_scratch_mp2_wipe_on_august_2016/descotea/coteharn/recobundles/results_se/'

        dname_whole_tracks = '/home/eleftherios/Data/Mammoth/mnt/parallel_scratch_mp2_wipe_on_august_2016/descotea/coteharn/recobundles/Hackethon_bdx/bordeaux_whole_brain_DTI_corrected/whole_brain_trks_60sj/'

        wb_trk1 = dname_whole_tracks + model_tag + ending_name
        static_basename = os.path.splitext(
            os.path.basename(wb_trk1))[0]

        current_tag = rec_tag

        wb_trk2 = dname_whole_tracks + current_tag + ending_name
        moving_basename = os.path.splitext(os.path.basename(wb_trk2))[0]
        moving_tag = moving_basename.split('_')[0]

        dname_results = out_folder + \
            subdir + bundle_type + '_slr_' + str(slr) + \
            '_prun_thr_' + str(pruning_thr) + '/'

        moved_streamlines_file = os.path.join(
            dname_results, moving_basename + '__to__' + static_basename + '.trk')

        dname_model_bundles = '/home/eleftherios/Data/Mammoth/mnt/parallel_scratch_mp2_wipe_on_august_2016/descotea/coteharn/recobundles/Hackethon_bdx/bordeaux_tracts_and_stems/'

        model_bundle_trk = dname_model_bundles + \
            model_tag + '/tracts/' + \
            bundle_type + '/' + model_tag + \
            '_' + bundle_type + '_GP.trk'

        base_sf = os.path.splitext(
            os.path.basename(moved_streamlines_file))[0]
        base_mb = os.path.splitext(os.path.basename(model_bundle_trk))[0]

        dname_results = out_folder + \
            subdir + bundle_type + '_slr_' + str(slr) + \
            '_prun_thr_' + str(pruning_thr) + '/'

        recognized_bundle_trk = os.path.join(
            dname_results,
            base_mb + '_of_' + base_sf + '.trk')

        expert_bundle_trk = dname_model_bundles + \
            moving_tag + '/tracts/' + \
            bundle_type + '/' + moving_basename.split('_')[0] + \
            '_' + bundle_type + '_GP.trk'


        model = nib.streamlines.load(model_bundle_trk).streamlines
        expert = nib.streamlines.load(expert_bundle_trk).streamlines
        auto = nib.streamlines.load(recognized_bundle_trk).streamlines

        return model, expert, auto

    all_bundles = []
    all_bundles_text = []

    for model_tag in df_imax.index[:12]:
        rec_tag = df_imax[model_tag]
        model_tag = model_tag.encode('ascii')
        rec_tag = rec_tag.encode('ascii')
        model, expert, auto = get_files(bundle_type, model_tag, rec_tag)
        all_bundles.append(model)
        all_bundles.append(expert)
        all_bundles.append(auto)
        all_bundles_text.append('model_' + model_tag)
        all_bundles_text.append('expert_' + rec_tag)
        all_bundles_text.append('auto_' + rec_tag)

    show_grid(all_bundles,
              all_bundles_text, dim=(3, 12),
              auto_orient=False, cell_padding=(-50, 50))

parser = IntrospectiveArgumentParser()
parser.add_workflow(cross_validation_flow)

if __name__ == '__main__':

    # args = parser.get_flow_args()
    # cross_validation_flow(**args)
    # main()
    # summarize_xlsx_mammoth()
    # final_table()
    find_argmax_and_show()
