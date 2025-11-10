import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
sys.path.append("/home/admin/wyb/DWT_CoMER")
# sys.path.append("/workspace/DWT_CoMER")

import typer
import zipfile
from comer.datamodule import CROHMEDatamodule
from comer.lit_comer import LitCoMER
from pytorch_lightning import Trainer, seed_everything

seed_everything(7)

def cal_distance(word1, word2):
    m = len(word1)
    n = len(word2)
    if m*n == 0:
        return m+n
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range (m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            a = dp[i-1][j] + 1
            b = dp[i][j-1] + 1
            c = dp[i-1][j-1]
            if word1[i-1] != word2[j-1]:
                c += 1
            dp[i][j] = min(a, b, c)
    return dp[m][n]


def main(version: str, test_year: str):
    # generate output latex in result.zip
    ckp_folder = os.path.join("lightning_logs", f"version_{version}", "checkpoints")
    fnames = os.listdir(ckp_folder)
    assert len(fnames) == 1
    ckp_path = os.path.join(ckp_folder, fnames[0])
    print(f"Test with fname: {fnames[0]}")

    trainer = Trainer(logger=False, gpus=1)

    dm = CROHMEDatamodule(test_year=test_year, eval_batch_size=4)
    
    model = LitCoMER.load_from_checkpoint(ckp_path)

    trainer.test(model, datamodule=dm)

    caption = {}
    with zipfile.ZipFile("data.zip") as archive:
        with archive.open(f"data/{test_year}/caption.txt", "r") as f:
            caption_lines = [line.decode('utf-8').strip() for line in f.readlines()]
            for caption_line in caption_lines:
                caption_parts = caption_line.split()
                caption_file_name = caption_parts[0]
                caption_string = ' '.join(caption_parts[1:])
                caption[caption_file_name] = caption_string
    
    # with zipfile.ZipFile("result.zip") as archive:
    #     exprate=[0,0,0,0]
    #     file_list = archive.namelist()
    #     txt_files = [file for file in file_list if file.endswith('.txt')]
    #     with open(os.path.join(ckp_folder, os.pardir, f"{test_year}_pred.txt"), "w") as pred_file:
    #         for txt_file in txt_files:
    #             file_name = txt_file.rstrip('.txt')
    #             with archive.open(txt_file) as f:
    #                 lines = f.readlines()
    #                 pred_string = lines[1].decode('utf-8').strip()[1:-1]
    #                 # 将 file_name 和 pred_string 写入新文件（每行一条）
    #                 pred_file.write(f"{file_name} {pred_string}\n")
    #                 if file_name in caption:
    #                     caption_string = caption[file_name]
    #                 else:
    #                     print(file_name,"not found in caption file")
    #                     continue
    #                 caption_parts = caption_string.strip().split()
    #                 pred_parts = pred_string.strip().split()
    #                 if caption_string == pred_string:
    #                     exprate[0]+=1
    #                 else:
    #                     # print(caption_string)
    #                     # print(pred_string)
    #                     error_num=cal_distance(pred_parts,caption_parts)
    #                     if error_num<=3:
    #                         exprate[error_num]+=1
    #     tot = len(txt_files)
    #     exprate_final=[]
    #     for i in range(1,5):
    #         exprate_final.append(100*sum(exprate[:i])/tot)
    #     # print(test_year,"exprate",exprate_final)
    #     with open(os.path.join(ckp_folder, os.pardir, f'{test_year}.txt'), 'w') as wf:
    #         wf.write(f'ExpRate:  {exprate_final[0]}\n')
    #         wf.write(f'ExpRate<=1:  {exprate_final[1]}\n')
    #         wf.write(f'ExpRate<=2:  {exprate_final[2]}\n')
    #         wf.write(f'ExpRate<=3:  {exprate_final[3]}\n')

    with zipfile.ZipFile("result.zip") as archive:
        exprate = [0, 0, 0, 0]
        file_list = archive.namelist()
        txt_files = [file for file in file_list if file.endswith('.txt')]
        
        # 同时打开预测结果文件和错误文件
        with open(os.path.join(ckp_folder, os.pardir, f"{test_year}_pred.txt"), "w") as pred_file, \
            open(os.path.join(ckp_folder, os.pardir, f"{test_year}_error.txt"), "w") as error_file:
            
            for txt_file in txt_files:
                file_name = txt_file.rstrip('.txt')
                with archive.open(txt_file) as f:
                    lines = f.readlines()
                    pred_string = lines[1].decode('utf-8').strip()[1:-1]
                    
                    # 写入预测结果文件（所有记录）
                    pred_file.write(f"{file_name} {pred_string}\n")
                    
                    if file_name in caption:
                        caption_string = caption[file_name]
                    else:
                        print(file_name, "not found in caption file")
                        continue
                    
                    # 比较预测结果
                    caption_parts = caption_string.strip().split()
                    pred_parts = pred_string.strip().split()
                    
                    if caption_string == pred_string:
                        exprate[0] += 1
                    else:
                        # 写入错误文件（仅预测错误的记录）
                        error_file.write(f"{file_name} {pred_string}\n")
                        
                        # 计算错误数量
                        error_num = cal_distance(pred_parts, caption_parts)
                        if error_num <= 3:
                            exprate[error_num] += 1

        # 后续统计和输出保持不变
        tot = len(txt_files)
        exprate_final = []
        for i in range(1, 5):
            exprate_final.append(100 * sum(exprate[:i]) / tot)
        
        with open(os.path.join(ckp_folder, os.pardir, f'{test_year}.txt'), 'w') as wf:
            wf.write(f'ExpRate:  {exprate_final[0]}\n')
            wf.write(f'ExpRate<=1:  {exprate_final[1]}\n')
            wf.write(f'ExpRate<=2:  {exprate_final[2]}\n')
            wf.write(f'ExpRate<=3:  {exprate_final[3]}\n')


if __name__ == "__main__":
    # main(98, "length_41-50")
    typer.run(main)
