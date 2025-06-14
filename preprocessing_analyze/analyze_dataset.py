import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

def analyze_dataset_distribution(csv_file_path):
    """
    分析資料集的分類分佈情況
    
    Args:
        csv_file_path (str): CSV檔案路徑
    """
    print("="*60)
    print("資料集分析報告")
    print("="*60)
    
    try:
        # 讀取資料集
        print("正在讀取資料集...")
        df = pd.read_csv(csv_file_path)
        
        # 基本資料集資訊
        print(f"\n📊 基本資料集資訊:")
        print(f"資料總筆數: {len(df):,}")
        print(f"欄位數量: {len(df.columns)}")
        print(f"欄位名稱: {list(df.columns)}")
        
        # 檢查是否有target欄位（常見的目標變數名稱）
        possible_target_columns = ['winner', 'label', 'target', 'class', 'winner_model_a', 'winner_model_b', 'tie']
        target_column = None
        
        for col in possible_target_columns:
            if col in df.columns:
                target_column = col
                break
        
        # 如果沒有找到明確的目標欄位，顯示所有欄位的唯一值
        if target_column is None:
            print(f"\n🔍 各欄位的唯一值數量:")
            for col in df.columns:
                unique_count = df[col].nunique()
                print(f"  {col}: {unique_count} 個唯一值")
                
                # 如果唯一值較少，顯示具體的值
                if unique_count <= 10:
                    unique_values = df[col].unique()
                    print(f"    值: {unique_values}")
        
        # 如果找到目標欄位，進行詳細分析
        if target_column:
            print(f"\n🎯 目標變數分析 (欄位: {target_column}):")
            value_counts = df[target_column].value_counts()
            percentages = df[target_column].value_counts(normalize=True) * 100
            
            print(f"\n類別分佈:")
            for category, count in value_counts.items():
                percentage = percentages[category]
                print(f"  {category}: {count:,} 筆 ({percentage:.2f}%)")
            
            # 檢查類別平衡性
            max_count = value_counts.max()
            min_count = value_counts.min()
            imbalance_ratio = max_count / min_count
            
            print(f"\n⚖️ 類別平衡性分析:")
            print(f"最多類別: {max_count:,} 筆")
            print(f"最少類別: {min_count:,} 筆")
            print(f"不平衡比例: {imbalance_ratio:.2f}:1")
            
            if imbalance_ratio > 2:
                print("⚠️  資料集存在類別不平衡問題")
            else:
                print("✅ 資料集類別分佈相對平衡")
            
            # 繪製分佈圖
            plt.figure(figsize=(12, 8))
            
            # 長條圖
            plt.subplot(2, 2, 1)
            value_counts.plot(kind='bar', color='skyblue', edgecolor='black')
            plt.title('類別分佈 - 絕對數量')
            plt.xlabel('類別')
            plt.ylabel('數量')
            plt.xticks(rotation=45)
            
            # 圓餅圖
            plt.subplot(2, 2, 2)
            plt.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
            plt.title('類別分佈 - 百分比')
            
            # 百分比長條圖
            plt.subplot(2, 2, 3)
            percentages.plot(kind='bar', color='lightcoral', edgecolor='black')
            plt.title('類別分佈 - 百分比')
            plt.xlabel('類別')
            plt.ylabel('百分比 (%)')
            plt.xticks(rotation=45)
            
            # 累積分佈圖
            plt.subplot(2, 2, 4)
            cumulative_percentages = percentages.cumsum()
            cumulative_percentages.plot(kind='line', marker='o', color='green', linewidth=2, markersize=6)
            plt.title('累積分佈')
            plt.xlabel('類別')
            plt.ylabel('累積百分比 (%)')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 儲存圖表
            output_dir = 'results'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            plt.savefig(f'{output_dir}/dataset_distribution_analysis.png', dpi=300, bbox_inches='tight')
            print(f"\n📈 圖表已儲存至: {output_dir}/dataset_distribution_analysis.png")
            plt.show()
        
        # 缺失值分析
        print(f"\n🔍 缺失值分析:")
        missing_data = df.isnull().sum()
        if missing_data.sum() == 0:
            print("✅ 沒有發現缺失值")
        else:
            print("發現缺失值:")
            for col, missing_count in missing_data[missing_data > 0].items():
                missing_percentage = (missing_count / len(df)) * 100
                print(f"  {col}: {missing_count} 個缺失值 ({missing_percentage:.2f}%)")
        
        # 資料型態分析
        print(f"\n📋 資料型態分析:")
        data_types = df.dtypes.value_counts()
        for dtype, count in data_types.items():
            print(f"  {dtype}: {count} 個欄位")
        
        # 儲存分析結果
        analysis_results = {
            '總筆數': len(df),
            '欄位數量': len(df.columns),
            '欄位名稱': list(df.columns),
            '缺失值總數': missing_data.sum(),
            '資料型態': dict(data_types)
        }
        
        if target_column:
            analysis_results['目標變數'] = target_column
            analysis_results['類別分佈'] = dict(value_counts)
            analysis_results['類別百分比'] = dict(percentages)
            analysis_results['不平衡比例'] = imbalance_ratio
        
        # 儲存到檔案
        import json
        with open(f'{output_dir}/dataset_analysis_results.json', 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 分析結果已儲存至: {output_dir}/dataset_analysis_results.json")
        
    except Exception as e:
        print(f"❌ 讀取檔案時發生錯誤: {e}")
        print("請確認檔案路徑是否正確，或檔案是否太大")

def analyze_specific_columns(csv_file_path, columns_to_analyze=None):
    """
    分析特定欄位的分佈
    
    Args:
        csv_file_path (str): CSV檔案路徑
        columns_to_analyze (list): 要分析的欄位列表
    """
    try:
        # 使用 chunksize 來處理大檔案
        print("正在分析大型檔案...")
        
        if columns_to_analyze is None:
            # 先讀取少量資料來確定欄位
            sample_df = pd.read_csv(csv_file_path, nrows=1000)
            print(f"樣本資料欄位: {list(sample_df.columns)}")
            return
        
        # 分塊讀取並統計
        chunk_size = 10000
        value_counts = {}
        total_rows = 0
        
        for chunk in pd.read_csv(csv_file_path, chunksize=chunk_size):
            total_rows += len(chunk)
            
            for col in columns_to_analyze:
                if col in chunk.columns:
                    if col not in value_counts:
                        value_counts[col] = Counter()
                    
                    # 統計當前chunk的值
                    chunk_counts = chunk[col].value_counts()
                    for value, count in chunk_counts.items():
                        value_counts[col][value] += count
            
            print(f"已處理 {total_rows:,} 筆資料...")
        
        # 顯示結果
        print(f"\n📊 總共處理了 {total_rows:,} 筆資料")
        
        for col, counts in value_counts.items():
            print(f"\n🎯 欄位 '{col}' 的分佈:")
            total_count = sum(counts.values())
            
            for value, count in counts.most_common():
                percentage = (count / total_count) * 100
                print(f"  {value}: {count:,} 筆 ({percentage:.2f}%)")
        
    except Exception as e:
        print(f"❌ 分析時發生錯誤: {e}")

if __name__ == "__main__":
    # 設定檔案路徑
    train_file = "train.csv"
    
    print("🚀 開始分析資料集...")
    
    # 檢查檔案是否存在
    if not os.path.exists(train_file):
        print(f"❌ 找不到檔案: {train_file}")
        print("請確認檔案是否在當前目錄中")
    else:
        file_size = os.path.getsize(train_file) / (1024 * 1024)  # MB
        print(f"📁 檔案大小: {file_size:.2f} MB")
        
        if file_size > 100:  # 如果檔案大於100MB
            print("📋 檔案較大，先查看欄位結構...")
            analyze_specific_columns(train_file)
              # 根據發現的欄位結構，分析模型比較的分類欄位
            classification_cols = ['winner_model_a', 'winner_model_b', 'winner_tie']
            print(f"\n🎯 分析模型勝負分類欄位: {classification_cols}")
            analyze_specific_columns(train_file, classification_cols)
        else:
            # 直接分析整個檔案
            analyze_dataset_distribution(train_file)
