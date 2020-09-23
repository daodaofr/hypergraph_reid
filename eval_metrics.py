from __future__ import print_function, absolute_import
import numpy as np
import copy

def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

def get_rank_list(dist_vec, q_id, q_cam, g_ids, g_cams, rank_list_size):
    sort_inds = np.argsort(dist_vec)
    rank_list = []
    same_id = []
    i = 0
    for ind, g_id, g_cam in zip(sort_inds, g_ids[sort_inds], g_cams[sort_inds]):
        # Skip gallery images with same id and same camera as query
        if (q_id == g_id) and (q_cam == g_cam):
            continue
        same_id.append(q_id == g_id)
        rank_list.append(ind)
        i += 1
        if i >= rank_list_size:
            break
    return rank_list, same_id

def save_rank_list_to_file(rank_list, same_id, g_file_path, save_path):
    g_imgs = []
    with open(g_file_path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:      
                break
            g_imgs.append(line.rstrip())
    
    fid = open(save_path, "a+")
    for ind, sid in zip(rank_list, same_id):
        fid.write(g_imgs[ind] + '\t')
        fid.write(str(sid) + '\t')
    fid.write('\n')
    fid.close()

def save_results(distmat, q_pids, g_pids, q_camids, g_camids):
    #save_path = "data/mars/mars_results_graphsage_new.txt"
    #query_file = "data/mars/mars_query.txt"
    #gallery_file = "data/mars/mars_gallery.txt"

    save_path = "data/prid2011/prid_results_graphsage_part.txt"
    query_file = "data/prid2011/prid_query.txt"
    gallery_file = "data/prid2011/prid_gallery.txt"
   
    #save_path = "data/ilids-vid/ilids_results_graphsage_part.txt"
    #query_file = "data/ilids-vid/ilids_query.txt"
    #gallery_file = "data/ilids-vid/ilids_gallery.txt"

    with open(save_path, "a+") as fid:
        fid.write("\n")
        fid.write("++++++++++++++++++++++++++++++++++++++++++\n")

    q_imgs = []
    with open(query_file, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            q_imgs.append(line.rstrip())

    for i in range(q_pids.shape[0]):
        rank_list, same_id = get_rank_list(
            distmat[i], q_pids[i], q_camids[i], g_pids, g_camids, 10)
        with open(save_path, "a+") as fid:
            fid.write(q_imgs[i] + ":\t")
        save_rank_list_to_file(rank_list, same_id, gallery_file, save_path) 
        
