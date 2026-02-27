import pandas as pd

def main():
    try:
        # Read only the header
        df = pd.read_csv('data/train.csv', nrows=0)
        columns = df.columns.tolist()
        
        with open('config/全部特征.txt', 'w', encoding='utf-8') as f:
            f.write("特征英文名,特征中文名,数据类型\n")
            for col in columns:
                # Use column name as Chinese name for now
                f.write(f"{col},{col},string\n")
        
        print(f"Generated 'config/全部特征.txt' with {len(columns)} features.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
