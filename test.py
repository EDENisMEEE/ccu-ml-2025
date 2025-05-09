import os
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from joblib import load
from features import extract_features, headerList
from fft_utils import FFT, FFT_data

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def data_generate():
    datapath = './test_data'
    tar_dir = 'tabular_data_test'
    pathlist_txt = Path(datapath).glob('**/*.txt')


    for file in tqdm(pathlist_txt, desc='Generating features'):
        f = open(file)

        All_data = []

        count = 0
        for line in f.readlines():
            if line == '\n' or count == 0:
                count += 1
                continue
            num = line.split(' ')
            if len(num) > 5:
                tmp_list = []
                for i in range(6):
                    tmp_list.append(int(num[i]))
                All_data.append(tmp_list)

        f.close()

        swing_index = np.linspace(0, len(All_data), 28, dtype = int)
        # filename.append(int(Path(file).stem))
        # all_swing.append([swing_index])


        with open('./{dir}/{fname}.csv'.format(dir = tar_dir, fname = Path(file).stem), 'w', newline = '') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headerList)
            try:
                a_fft, g_fft = FFT_data(All_data, swing_index)
                a_fft_imag = [0] * len(a_fft)
                g_fft_imag = [0] * len(g_fft)
                n_fft, a_fft, a_fft_imag = FFT(a_fft, a_fft_imag)
                n_fft, g_fft, g_fft_imag = FFT(g_fft, g_fft_imag)
                for i in range(len(swing_index)):
                    if i==0:
                        continue
                    features = extract_features(All_data[swing_index[i-1]: swing_index[i]], i - 1, len(swing_index) - 1, n_fft, a_fft, g_fft, a_fft_imag, g_fft_imag)
                    writer.writerow(features)
            except:
                print(Path(file).stem)
                continue


def main():
    # 若尚未產生特徵，請先執行 data_generate() 生成特徵 CSV 檔案
    # data_generate()

    submission = pd.read_csv("sample_submission.csv")
    # 讀取訓練資訊，根據 player_id 將資料分成 80% 訓練、20% 測試
    info = pd.read_csv("test_info.csv")
    # unique_players = info['unique_id'].unique()
    # train_players, test_players = train_test_split(unique_players, test_size=0.2, random_state=42)

    # 讀取特徵 CSV 檔（位於 "./tabular_data_train"）
    datapath = './tabular_data_test'
    # datalist = list(Path(datapath).glob('**/*.csv'))
    target_info = {
        'gender': {'task_type': 'binary', 'columns': ['gender']},
        'hold': {'task_type': 'binary', 'columns': ['hold racket handed']},
        'years': {
            'task_type': 'multiary',
            'columns': ['play years_0', 'play years_1', 'play years_2']
        },
        'levels': {
            'task_type': 'multiary',
            'columns': ['level_2', 'level_3', 'level_4', 'level_5']
        }
    }


    scaler = load(f'./models/scaler.joblib')
    encoder = load(f'./models/le.joblib')

    for target, config in target_info.items():
        pred_matrix = []
        print(f"Predicting {target}...")
        model = load(f'./models/xgb_{target}_model.joblib')

        task_type = config['task_type']
        columns = config['columns']

        predictions = []
        for uid in submission['unique_id']:
            data_path = Path(f"{datapath}/{uid}.csv")
            if not data_path.exists():
                predictions.append(encoder.classes_[0])  # fallback to default
                continue

            df = pd.read_csv(data_path)
            X_scaled = scaler.transform(df)


            if task_type == 'binary':
                probs = model.predict_proba(X_scaled)
                avg_prob = np.mean([p[1] for p in probs])  # positive class (e.g., male or right-hand)
                avg_prob = 1 - avg_prob # reverse prob
                # print(f"probal = {avg_prob}")
                pred_matrix.append([avg_prob])

            else:
                probs = model.predict_proba(X_scaled)  # shape: (27, num_classes)
                class_sums = np.sum(probs, axis=0)  # sum over 27 predictions
                class_avg = class_sums / len(probs)  # linear average of total sums

                aligned_probs = [0.0] * len(columns)
                for i, cls in enumerate(columns):
                    cls_str = str(cls)
                    if f"{cls_str}" in columns:
                        idx = columns.index(f"{cls_str}")
                        aligned_probs[idx] = class_avg[i]
                # 驗證機率總和為 1
                s = sum(aligned_probs)
                if not np.isclose(s, 1.0):
                    if s > 0:
                        aligned_probs = [x / s for x in aligned_probs]
                    else:
                        # 如果總和為 0（極端錯誤），分配均勻機率
                        aligned_probs = [1.0 / len(aligned_probs)] * len(aligned_probs)
                # print(f"probal = {avg_prob}")
                pred_matrix.append(aligned_probs)

        assert len(submission) == len(pred_matrix)
        for i, col in enumerate(columns):
            submission[col] = [row[i] for row in pred_matrix]
    print("Writing csv file...")
    submission.to_csv("submission.csv",float_format="%.6f", index=False)
    print("✅ 預測完成，已儲存為 submission.csv")


if __name__ == '__main__':
    main()
