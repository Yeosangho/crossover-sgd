import json
import csv
f = open("/scratch/x2026a02/baseline_fujitsu/result_1.txt", "r")
summary = open('/scratch/x2026a02/baseline_fujitsu/summary1.csv','w')
wr = csv.writer(summary)
while True : 
	line = f.readline()
	if not line : break
	if("eval_accuracy" in line):
		print(line)
		print(line.index('{'))
		line_dict = json.loads(line[line.index('{'):])
		print(line_dict['value'])
		print(line_dict['metadata']["epoch_num"])
		wr.writerow([line_dict['metadata']["epoch_num"], line_dict['value']])
summary.close()
f.close()