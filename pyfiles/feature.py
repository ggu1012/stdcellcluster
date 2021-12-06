import numpy as np
import json

con_file = open('modfiles/connectivity.json', 'r')
mem_file = open('modfiles/memory_level.json', 'w')

conn = json.load(con_file)
cells = list(conn.keys())

###### memory level

# Find starting points - memory macros
start = []
for cell in cells:
    if 'sram' in cell:
        start.append(cell)

memory_level = dict()


for cell in cells:
    memory_level[cell] = [-1 for n in range(len(start))]


# BFS
for idx, cell in enumerate(start):

    stack = np.array([cell], dtype=object)

    level = 0
    visited = np.array([], dtype=object)

    while visited.size != len(conn):        

        stack_new = []

        for cell in stack:
            if cell in visited:
                stack = np.delete(stack, np.where(stack == cell))

        visited = np.append(visited, stack)

        for cell in stack:
            memory_level[cell][idx] = level

            rm_connectivity = []
            for one in conn[cell]:
                splitted = one.split('/')
                joined = '/'.join(splitted[:-1])
                rm_connectivity.append(joined)

            neighbor = np.array(rm_connectivity, dtype=object)

            for one in neighbor:
                 if memory_level[one][idx] != -1:
                    neighbor = np.delete(neighbor, np.where(neighbor == one))

            stack_new = np.unique(np.append(stack_new, neighbor))

        level += 1

        stack = stack_new

# dict to np array
memory_level = np.array(list(memory_level.values()), dtype=object)

# normalize
memory_level = memory_level / memory_level.max()


###### hierarchy 

hier_list = cells[0].split('/')
max_length = len(hier_list)

inst_type = dict()

num = 0

for hier in hier_list:
    tmp = len(inst_type)
    inst_type[hier] = num
    num += 1

hier_feat = [inst_type[hier] if hier in inst_type else 0 for hier in hier_list]


for inst in cells[1:]:
    hier_list = inst.split('/')
    hier_length = len(hier_list)

    for hier in hier_list:
        if hier not in inst_type:
            inst_type[hier] = num
            num += 1
                  

    if hier_length > max_length:            
        hier_feat = np.pad(hier_feat, ((0,0),(0, hier_length-max_length))) if max_length != 1 \
            else np.pad(hier_feat, (0, hier_length-max_length))

        max_length = hier_length

    elif hier_length < max_length:
        hier_list = np.pad(hier_list, (0, max_length-hier_length))

    mapped_hier = [inst_type[hier] if hier in inst_type else 0 for hier in hier_list]

    hier_feat = np.vstack((hier_feat, mapped_hier))

# normalize
normalized_feat = hier_feat / hier_feat.max(axis=0)


feature = np.hstack((normalized_feat, memory_level))
np.save('modfiles/feature.npy', feature)


con_file.close()
mem_file.close()






        

