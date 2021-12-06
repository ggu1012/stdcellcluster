import os
from pyparsing import *
import json

netlist = open(os.getcwd() + "/outputs/flat_syn.v", "r")
instance_info = open('final_reports/instance.txt', 'r')
finst_area = open(os.getcwd() + "/modfiles/inst_area.json", "w")


line = netlist.readline()

chunk = []
WIREFLAG = False

parentheses = Word('(' + ')')
wire = Word('\\' + alphanums + '/' + '_')
parsing = parentheses + wire('wire') + parentheses





# 앞에 쓸데 없는 line 제거, module 부터 읽도록
for n in range(6):
    line = netlist.readline()

inst_type = dict()

while True:
    line = netlist.readline().lstrip(" ")

    if not line:
        break

    if "wire" in line:
        WIREFLAG = True

    if WIREFLAG == True and ";" in line:
        WIREFLAG = False
        continue

    if not WIREFLAG:
        chunk += line.rstrip("\n")

        if ";" in line:
            chunk = "".join(chunk)

            if (
                "module" not in chunk
                and "input" not in chunk
                and "output" not in chunk
                and "assign" not in chunk
            ):

                subchunk = chunk.split(' ')
                cellType = subchunk[0] 
                instance = subchunk[1] if '\\' not in subchunk[1] else subchunk[1][1:]

                inst_type[instance] = cellType

            chunk = []


inst_line = instance_info.readline()
while '-----' not in inst_line:
    inst_line = instance_info.readline()

cell_area = dict()

while True:
    cell_line = instance_info.readline()

    if '-----' in cell_line:
        break

    cell_line = cell_line.split(' ')

    area_line = instance_info.readline()
    area_line = area_line.lstrip(' ').split(' ')
    
    cell_area[cell_line[0]] = float(area_line[0])




inst_area = dict()
for cell in inst_type.keys():
    inst_area[cell] = cell_area[inst_type[cell]]

finst_area.write(json.dumps(inst_area))

netlist.close()
instance_info.close()
finst_area.close()
