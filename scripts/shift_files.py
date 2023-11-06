import shutil
import os

with open(snakemake.input[0],'r') as f:
    locations=f.read().split('\n')

for i in range(1,len(locations)-1):
    shutil.move(locations[i],locations[0])

with open(locations[0]+'csv_paths.txt','a') as f:
    f.writelines([locations[0]+s[s.rfind('/')+1:]+'\n' for s in locations[1:-2]])

with open(snakemake.output[0],'a') as f:
    f.writelines(['Bayesian Optimization performed successfully!\n','Check '+locations[0]+' for output files'])

with open(snakemake.output[1],'a') as f:
    f.write('Temporary file')
