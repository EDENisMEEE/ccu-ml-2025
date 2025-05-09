from features import extract_features, headerList
from fft_utils import FFT, FFT_data
from pathlib import Path
import numpy as np
import pandas as pd
import csv
import os
from joblib import dump #儲存權重與模型
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from model import model_binary_knn, model_multiary_knn, model_binary, model_multiary, model_binary_xgb, model_multiary_xgb


def data_generate():
    datapath = './train_data'
    tar_dir = 'tabular_data_train'
    pathlist_txt = Path(datapath).glob('**/*.txt')
    os.makedirs(tar_dir, exist_ok= True)


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
    datapath = './tabular_data_train'
    if Path(datapath).exists() == False:
        print("Generating training data...")
        data_generate()

    print("Preparing training X and Y...")
    # 讀取訓練資訊，根據 player_id 將資料分成 80% 訓練、20% 測試
    info = pd.read_csv('train_info.csv')
    unique_players = info['player_id'].unique()
    # train_players, test_players = train_test_split(unique_players, test_size=0.2, random_state=42)

    # 讀取特徵 CSV 檔（位於 "./tabular_data_train"）
    
    datalist = list(Path(datapath).glob('**/*.csv'))
    target_mask = ['gender', 'hold racket handed', 'play years', 'level']

    # 根據 test_players 分組資料
    x_train = pd.DataFrame()
    y_train = pd.DataFrame(columns=target_mask)
    # x_test = pd.DataFrame()
    # y_test = pd.DataFrame(columns=target_mask)

    for file in datalist:
        unique_id = int(Path(file).stem)
        row = info[info['unique_id'] == unique_id]
        if row.empty:
            continue
        player_id = row['player_id'].iloc[0]
        data = pd.read_csv(file)
        target = row[target_mask]
        target_repeated = pd.concat([target] * len(data))
        x_train = pd.concat([x_train, data], ignore_index=True)
        y_train = pd.concat([y_train, target_repeated], ignore_index=True)
        #     y_train = pd.concat([y_train, target_repeated], ignore_index=True)
        # if player_id in train_players:
        #     x_train = pd.concat([x_train, data], ignore_index=True)
        #     y_train = pd.concat([y_train, target_repeated], ignore_index=True)
        # elif player_id in test_players:
        #     x_test = pd.concat([x_test, data], ignore_index=True)
        #     y_test = pd.concat([y_test, target_repeated], ignore_index=True)

    # 標準化特徵
    scaler = MinMaxScaler()
    le = LabelEncoder()
    X_train_scaled = scaler.fit_transform(x_train)
    # X_test_scaled = scaler.transform(x_test)

    group_size = 27 #27次揮拍



    # 評分：針對各目標進行模型訓練與評分
    y_train_le_gender = le.fit_transform(y_train['gender'])
    # y_test_le_gender = le.transform(y_test['gender'])
    # model_binary_xgb(X_train_scaled, y_train_le_gender, X_test_scaled, y_test_le_gender, group_size, 'gender')
    model_binary_xgb(X_train_scaled, y_train_le_gender, 'gender')

    y_train_le_hold = le.fit_transform(y_train['hold racket handed'])
    # y_test_le_hold = le.transform(y_test['hold racket handed'])
    # model_binary_xgb(X_train_scaled, y_train_le_hold, X_test_scaled, y_test_le_hold, group_size, 'hold')
    model_binary_xgb(X_train_scaled, y_train_le_hold, 'hold')

    y_train_le_years = le.fit_transform(y_train['play years'])
    # y_test_le_years = le.transform(y_test['play years'])
    # model_multiary_xgb(X_train_scaled, y_train_le_years, X_test_scaled, y_test_le_years, group_size, 'years')
    model_multiary_xgb(X_train_scaled, y_train_le_years, 'years')

    y_train_le_level = le.fit_transform(y_train['level'])
    # y_test_le_level = le.transform(y_test['level'])
    # model_multiary_xgb(X_train_scaled, y_train_le_level, X_test_scaled, y_test_le_level, group_size, 'levels')
    model_multiary_xgb(X_train_scaled, y_train_le_level, 'levels')

    #AUC SCORE: 0.792(gender) + 0.998(hold) + 0.660(years) + 0.822(levels)

    dump(scaler, './models/scaler.joblib')
    dump(le, './models/le.joblib')

if __name__ == '__main__':
    main()
