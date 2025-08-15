import pandas as pd
import numpy as np
import os


def process_excel_file(file_path,output_path):
    # 读取Excel文件
    xls = pd.ExcelFile(file_path)
    sheet_names = xls.sheet_names

    # 创建新的Excel writer

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:

        for sheet_name in sheet_names:
            # 读取sheet为DataFrame（不自动识别表头）
            df = pd.read_excel(xls, sheet_name=sheet_name, header=None)

            # # 1. 处理封面页
            # if sheet_name == '封面':
            #     if not df.empty:
            #         # 在第一列左侧插入空列
            #         df.insert(0, '新列', np.nan)
            #     df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
            #     continue
            # 1. 处理封面页
            if sheet_name == '封面':
                if not df.empty:
                    # 创建一个新的空行（与df列数相同）
                    empty_row = pd.DataFrame([[''] * len(df.columns)], columns=df.columns)

                    # 将空行添加到DataFrame的顶部
                    df = pd.concat([empty_row, df], ignore_index=True)

                df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
                continue

            # 2. 处理流程图表格
            if not df.empty:
                # 检查是否包含目标列名
                found_columns = False
                header_row = None

                for i, row in df.iterrows():
                    row_str = ' '.join(row.dropna().astype(str))
                    if '步骤' in row_str and '工作流程' in row_str:
                        found_columns = True
                        header_row = i
                        break

                if found_columns and header_row is not None:
                    # 表头治理 - 删除说明行，设置新表头
                    first_row = df.iloc[0].values.tolist()
                    print("**********************************")
                    tmp_sheet_name = first_row[0]
                    tmp_sheet_name = tmp_sheet_name.replace('/', '-')
                    tmp_sheet_name = tmp_sheet_name.replace(' ', '')


                    new_header = df.iloc[header_row].fillna('').astype(str)
                    df = df.iloc[header_row + 1:]
                    df.columns = new_header
                    if "." in tmp_sheet_name:
                        cur_sheet_name = sheet_name
                    else:
                        cur_sheet_name = sheet_name + '_' + tmp_sheet_name

                    # 写入治理后的sheet
                    df.to_excel(writer, sheet_name=cur_sheet_name, index=False)

                    # 创建转置sheet
                    transposed_df = df.transpose()
                    transposed_sheet_name = f"{sheet_name}_转置"
                    transposed_df.to_excel(
                        writer,
                        sheet_name=transposed_sheet_name,
                        index=True,
                        header=False
                    )
                    continue

            # 3. 其他sheet直接复制
            df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)

    print(f"文件处理完成，已保存至: {output_path}")


def get_all_file_names(folder_path):
    """
    获取文件夹下所有文件的名称

    参数:
    folder_path: 文件夹路径
    """
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在")
        return []

    # 检查是否为文件夹
    if not os.path.isdir(folder_path):
        print(f"错误：'{folder_path}' 不是文件夹")
        return []

    # 获取所有文件名称
    file_names = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            file_names.append(item)

    # 如果没有找到文件
    if not file_names:
        print(f"警告：文件夹 '{folder_path}' 中没有找到任何文件")
        return []

    return file_names


def print_file_names(file_names):
    """
    打印文件名称列表

    参数:
    file_names: 文件名称列表
    """
    if not file_names:
        print("没有找到文件")
        return

    print(f"找到 {len(file_names)} 个文件:")
    for i, file_name in enumerate(file_names, 1):
        if "~$" in file_name:
            continue
        print(f"{i}. {file_name}")


def handle_all_wos_datas(folder_path,output_folder_path):

    file_names = get_all_file_names(folder_path)
    if not file_names:
        print("没有找到文件")
        return

    print(f"找到 {len(file_names)} 个文件:")

    for i, file_name in enumerate(file_names, 1):
        if "~$" in file_name:
            continue
        if file_name.endswith(".xls"):
            continue
        print(f"{i}. {file_name}")

        file_path = folder_path + "/" + file_name
        print("**"*58)
        output_file_name = file_name.replace(".xlsx","")
        # output_file_name = output_file_name.replace(".xls", "")

        output_path = output_folder_path + "/" + output_file_name + '_processed.xlsx'
        print(file_path)
        print(output_path)

        process_excel_file(file_path,output_path)



# 使用示例
if __name__ == "__main__":
    # input_file = "原始数据/QGWDG.C612-2018人力资源规划与计划控制程序.xlsx"  # 替换为实际文件路径
    # process_excel_file(input_file)

    # 设置要扫描的文件夹路径
    folder_path = "原始数据"  # 替换为实际文件夹路径
    output_folder_path = "自动治理数据"
    # 获取并打印文件名称

    handle_all_wos_datas(folder_path,output_folder_path)