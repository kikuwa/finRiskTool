import json
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse
from datetime import datetime

# --- 模块路径配置 ---
from .rules import RuleBase


def process_single_file(file_path: str, output_dir: str):
    """
    处理单个 JSONL 文件，输出 good/bad/index 文件。
    """
    file_path = Path(file_path)
    output_dir = Path(output_dir)

    # 读取输入文件
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        input_data = [json.loads(line.strip()) for line in lines if line.strip()]
    except Exception as e:
        print(f"Error reading or parsing {file_path}: {e}", file=sys.stderr)
        return None, None

    # 输出文件名（保留原文件名前缀）
    stem = file_path.stem
    good_output_file = output_dir / f'good_{stem}.jsonl'
    bad_output_file = output_dir / f'bad_{stem}.jsonl'
    bad_index_output_file = output_dir / f'index_of_bad_{stem}.jsonl'

    # 初始化 RuleBase（无状态，可复用）
    rule_base = RuleBase()

    # 打开输出文件（一次性写入模式）
    with open(good_output_file, 'w', encoding='utf-8') as good_f, \
         open(bad_output_file, 'w', encoding='utf-8') as bad_f, \
         open(bad_index_output_file, 'w', encoding='utf-8') as index_f:

        for idx, data in enumerate(input_data):
            try:
                # 提取字段
                user_content = (data.get('instruction', '') + ' ' + data.get('input', '')).strip()
                assistant_content_pri = data.get('output', "")
                ref_answer = data.get('gt', '')
                meta_prompt = data.get('meta_prompt', '你生成综合质量(有用性，事实性，无害性)极好的回复')

                # 安全地添加 think 标签（仅当明显缺失时）
                if not ('<think>' in assistant_content_pri or 'think>' in assistant_content_pri):
                    assistant_content = '<think>\n</think>\n' + assistant_content_pri
                else:
                    assistant_content = assistant_content_pri

                # 执行规则检查
                check_result = rule_base.run({
                    'meta_prompt': meta_prompt,
                    'user': user_content,
                    'assistant': assistant_content,
                    'file_path': str(file_path),
                    'ref_answer': ref_answer
                })

                # 判断是否全部通过
                bool_fields = {k: v for k, v in check_result.items() if isinstance(v, bool)}
                all_true = all(bool_fields.values())

                # 正确处理 index 字段（0 是合法值）
                index_end = data.get('index', idx)

                if all_true:
                    good_entry = {
                        'instruction': data.get('instruction', ''),
                        'input': data.get('input', ''),
                        'output': assistant_content,
                        'gt': ref_answer,
                        'index': index_end
                    }
                    good_f.write(json.dumps(good_entry, ensure_ascii=False) + '\n')
                else:
                    bad_entry = data.copy()
                    bad_entry['reason'] = check_result.get('warning', 'Unknown rule violation')
                    bad_entry['index'] = index_end
                    bad_f.write(json.dumps(bad_entry, ensure_ascii=False) + '\n')

                    bad_index = {
                        'index': index_end,
                        'reason': check_result.get('warning', 'Unknown rule violation')
                    }
                    index_f.write(json.dumps(bad_index, ensure_ascii=False) + '\n')

            except Exception as e:
                print(f"Error processing line {idx} in {file_path}: {e}", file=sys.stderr)
                continue

    return str(good_output_file), str(bad_output_file)


def main(input_data_path: str, output_dir: str, single_or_parallel: str = 'single'):
    start_time = datetime.now()
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    input_path = Path(input_data_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 收集所有 .jsonl 文件
    if input_path.is_file():
        if input_path.suffix == '.jsonl':
            files = [str(input_path)]
        else:
            print(f"Warning: Input file {input_path} is not a .jsonl file.", file=sys.stderr)
            return
    elif input_path.is_dir():
        files = [str(f) for f in input_path.rglob('*.jsonl')]
    else:
        print(f"Error: Input path {input_data_path} does not exist.", file=sys.stderr)
        return

    print(f"Found {len(files)} JSONL file(s) to process.")

    if single_or_parallel == 'single':
        print("Running in SINGLE-thread mode...")
        for file in files:
            print(f"Processing: {file}")
            process_single_file(file, output_dir)
    else:
        print("Running in PARALLEL mode...")
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_single_file, file, output_dir) for file in files]
            for future in futures:
                result = future.result()
                if result[0]:
                    print(f"Completed: {result}")

    end_time = datetime.now()
    total_duration = end_time - start_time
    print("\n====================")
    print("All done!")
    print(f"Total Time: {total_duration}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run data quality inspection on JSONL files.")
    parser.add_argument('--input_data_path', type=str, required=True,
                        help="Path to a JSONL file or directory containing JSONL files")
    parser.add_argument('--output_dir', type=str, default='output',
                        help="Output directory for processed files")
    parser.add_argument('--single_or_parallel', type=str, default='single',
                        choices=['single', 'parallel'],
                        help="Execution mode")

    args = parser.parse_args()
    main(args.input_data_path, args.output_dir, args.single_or_parallel)