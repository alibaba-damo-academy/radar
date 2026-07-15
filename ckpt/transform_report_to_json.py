import os
import pandas as pd
import json


report_path = './download/merlinabdominalctdataset/reports_final.xlsx'
info = pd.read_excel(report_path)

info_json = {}
for pid,report,split,fewshot in zip(info['study id'], info['Findings'], info['Split'], info['Few Shot']):
    if 'IMPRESSION:' in report:
        imp_idx = report.find('IMPRESSION:')
        findings = report[:imp_idx]
        impression = report[imp_idx:]
    else:
        findings = report
        impression = ''
    
    if pid not in info_json:
        info_json[pid] = {'report': report, 'findings': findings, 'impression': impression, 'split': split, 'fewshot': fewshot}
    else:
        print(pid)
        # print('exist', info_json[pid]['report'])
        # print('now', report)


save_path = './ckpt/merlin_report.json'
json.dump(info_json, open(save_path, 'w'), indent=4, ensure_ascii=False)

print('done')


