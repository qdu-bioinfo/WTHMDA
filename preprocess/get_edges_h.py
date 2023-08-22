#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd  
import numpy as np
import math
from ete3 import NCBITaxa      



def get_edges_h(MicroPd_path):
    # In[ ]:


    ncbi = NCBITaxa()  
    ncbi.update_taxonomy_database()   


    # In[ ]:
    print('get_edges_h start')


    MicroPd = pd.read_excel(MicroPd_path,header=None )
    #MicroPd = pd.read_excel('data/HMDAD/microbe_list.xlsx',header=None )
    MicroPd.columns = ['num_id', 'M_name']
    M_size = MicroPd.shape[0]


    # In[ ]:


    def find_ncbi_id(name):
            ncbi_id = ncbi.get_name_translator([name])
            if ncbi_id=={}:
                return 0
            else:
                return list(ncbi_id.values())[0][0]
    def find_lineage(ncbi_id):
            ncbi_lineage = ncbi.get_lineage(ncbi_id)
            if ncbi_lineage==None:
                return 0
            else:
                return ncbi_lineage
    MicroPd['ncbi_id'] = MicroPd.apply(lambda x : find_ncbi_id(x.M_name), axis=1)
    MicroPd['lineage'] = MicroPd.apply(lambda x : find_lineage(x.ncbi_id), axis=1)
    higher = []
    lower = []
    for i in range(len(MicroPd)):
        lower_tax_id = MicroPd.iloc[i, 2]
        if lower_tax_id == 0:
            continue
        lower_id = MicroPd.iloc[i, 0]
        lineage = MicroPd.iloc[i, 3]
        for j in range((len(lineage) - 2), -1, -1):
            next_point = lineage[j]
            if next_point in list(MicroPd['ncbi_id']):
                higher.append(int(MicroPd[MicroPd['ncbi_id']==next_point]['num_id']))
                lower.append(lower_id)
                break
    ha = list(np.arange(0,M_size))
    la = list(np.arange(0,M_size))
    higher_sl = higher + ha
    lower_sl = lower + la
    selfLoop_tree = pd.DataFrame(data={'higher' : higher_sl, 'lower' : lower_sl})


    # In[ ]:


    base_init = pd.DataFrame(data = {'micro_A' : np.zeros(M_size*M_size), 'micro_B' : np.zeros(M_size*M_size)})
    for i in range(M_size):
        for j in range(M_size):
            base_init.iloc[(i*M_size)+j][0] = i
            base_init.iloc[(i*M_size)+j][1] = j
    def get_tax_id(num_id, MicroPd):
        tax_id = int(MicroPd[MicroPd['num_id']==num_id]['ncbi_id'])
        return tax_id
    base_init['A_tax_id'] = base_init.apply(lambda x : get_tax_id(x.micro_A, MicroPd), axis=1)
    base_init['B_tax_id'] = base_init.apply(lambda x : get_tax_id(x.micro_B, MicroPd), axis=1)


    # In[ ]:


    WP_init = base_init.copy(deep=True)
    def get_micro_depth(tax_id, MicroPd):
        if tax_id==0:
            return 0
        else:
            micro_depth = len(ncbi.get_lineage(tax_id))
            return micro_depth   
    WP_init['A_depth'] = WP_init.apply(lambda x : get_micro_depth(x.A_tax_id, MicroPd), axis=1)
    WP_init['B_depth'] = WP_init.apply(lambda x : get_micro_depth(x.B_tax_id, MicroPd), axis=1)
    def get_common_ancestor_depth(A_tax_id, B_tax_id, tree):
        if A_tax_id==0 or B_tax_id==0:
            return 0
        elif A_tax_id == B_tax_id:
            return len(ncbi.get_lineage(A_tax_id))
        else:
            common_ancestor = tree.get_common_ancestor(str(int(A_tax_id)), str(int(B_tax_id)))
            ca_tax_id = common_ancestor.taxid
            ca_depth = len(ncbi.get_lineage(ca_tax_id))
            return ca_depth
    tax_id_list = list(MicroPd[MicroPd['ncbi_id']!=0]['ncbi_id'])
    tree = ncbi.get_topology(tax_id_list)
    WP_init['CA_depth'] = WP_init.apply(lambda x : get_common_ancestor_depth(x.A_tax_id, x.B_tax_id,tree), axis=1)
    def get_WP(A_depth, B_depth, CA_depth):
        if CA_depth==0:
            return 0
        else:
            WP = (2*CA_depth)/(A_depth + B_depth)
            return WP
    WP_init['WP'] = WP_init.apply(lambda x : get_WP(x.A_depth, x.B_depth, x.CA_depth), axis=1)


    # In[ ]:


    LC_init = base_init.copy(deep=True)
    def find_lineage_len(ncbi_id):
            ncbi_lineage = ncbi.get_lineage(ncbi_id)
            if ncbi_lineage==None:
                return 0
            else:
                len_lineage = len(ncbi_lineage)
                return len_lineage
    MicroPd['len_lineage'] = MicroPd.apply(lambda x : find_lineage_len(x.ncbi_id), axis=1)
    MicroPd


    # In[ ]:


    depth_num = int(MicroPd['len_lineage'].max())
    max_depth = depth_num
    tax_id_list = list(MicroPd[MicroPd['ncbi_id']!=0]['ncbi_id'])
    tree = ncbi.get_topology(tax_id_list)
    def get_dist(A_tax_id, B_tax_id, tree):
        if A_tax_id==0 or B_tax_id==0:
            return 99999
        elif A_tax_id == B_tax_id:
            return 0
        else:
            AB_dist = tree.get_distance(str(int(A_tax_id)), str(int(B_tax_id)))
            return AB_dist
    LC_init['AB_dist'] = LC_init.apply(lambda x : get_dist(x.A_tax_id, x.B_tax_id, tree), axis=1)


    # In[ ]:


    def get_LC(AB_dist, max_depth):
        if AB_dist==99999:
            return 0
        else:
            LC = 1 - ((math.log2(AB_dist + 1))/(math.log2((2 * max_depth) + 1)))
            return LC
    LC_init['LC'] = LC_init.apply(lambda x : get_LC(x.AB_dist, max_depth), axis=1)


    # In[ ]:


    N = len(MicroPd[MicroPd['ncbi_id']!=0])
    count_v_d = {}
    for i in range(M_size):
        v_list = MicroPd.iloc[i,3]
        if v_list == 0:
            continue
        for j in range(len(v_list)):
            if v_list[j] not in count_v_d.keys():
                count_v_dict = count_v_d.update({v_list[j] : 1})
            else:
                count_v_d[v_list[j]] += 1


    # In[ ]:


    Lin_init = base_init.copy(deep=True)
    def get_IC(tax_id, count_v_d, N):
        if tax_id==0:
            return 0
        else:
            count_v = count_v_d[tax_id]
            pv = count_v / N
            IC = -math.log2(pv)
            return IC
    Lin_init['A_IC'] = Lin_init.apply(lambda x : get_IC(x.A_tax_id, count_v_d, N), axis=1)
    Lin_init['B_IC'] = Lin_init.apply(lambda x : get_IC(x.B_tax_id, count_v_d, N), axis=1)
    tax_id_list = list(MicroPd[MicroPd['ncbi_id']!=0]['ncbi_id'])
    tree = ncbi.get_topology(tax_id_list)
    def get_CA_IC(A_tax_id,B_tax_id, count_v_d, N, tree):
        if A_tax_id==0 or B_tax_id==0:
            return 0
        elif A_tax_id == B_tax_id:
            count_v = count_v_d[A_tax_id]
            pv = count_v / N
            A_IC = -math.log2(pv)
            return A_IC
        else:
            common_ancestor = tree.get_common_ancestor(str(int(A_tax_id)), str(int(B_tax_id)))
            ca_tax_id = common_ancestor.taxid
            count_v = count_v_d[ca_tax_id]
            pv = count_v / N
            IC = -math.log2(pv)
            return IC
    Lin_init['CA_IC'] = Lin_init.apply(lambda x : get_CA_IC(x.A_tax_id, x.B_tax_id, count_v_d, N, tree), axis=1)


    # In[ ]:


    JC_init = Lin_init.copy(deep=True)
    def get_Lin(A_IC, B_IC, CA_IC):
        if A_IC==0 or B_IC==0 or CA_IC==0:
            return 0
        else:
            sim = (2 * CA_IC) / (A_IC + B_IC)
            return sim
    Lin_init['Lin'] = Lin_init.apply(lambda x : get_Lin(x.A_IC, x.B_IC, x.CA_IC), axis=1)


    # In[ ]:


    def get_JC(A_IC, B_IC, CA_IC, la):
        dist = (A_IC + B_IC) - (2 * CA_IC)
        sim = math.exp(-(dist / la))
        return sim
    JC_init['JC'] = JC_init.apply(lambda x : get_JC(x.A_IC, x.B_IC, x.CA_IC, la = 5), axis=1)


    # In[ ]:


    WP_W = WP_init.drop(['A_tax_id','B_tax_id','A_depth', 'B_depth', 'CA_depth'], axis=1)
    LC_W = LC_init.drop(['A_tax_id','B_tax_id', 'AB_dist'], axis=1)
    Lin_W = Lin_init.drop(['A_tax_id','B_tax_id','A_IC', 'B_IC', 'CA_IC'], axis=1)
    JC_W = JC_init.drop(['A_tax_id','B_tax_id','A_IC', 'B_IC', 'CA_IC'], axis=1)
    selfLoop_tree.to_csv('data/HMDAD/tree_edge.csv', index=False)
    LC_W.to_csv('data/HMDAD/LC_edge.csv', index=False)
    WP_W.to_csv('data/HMDAD/WP_edge.csv', index=False)
    Lin_W.to_csv('data/HMDAD/Lin_edge.csv', index=False)
    JC_W.to_csv('data/HMDAD/JC_edge.csv', index=False)
    print('get_edges_h done')

if __name__ == "__main__":
    MicroPd_path = 'data/HMDAD/microbe_list.xlsx'
    get_edges_h(MicroPd_path)
    print('get_edges_h done')