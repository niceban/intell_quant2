import os
import datetime
from pathlib import Path

def get_folder_structure(path, prefix="", is_last=True, level=0):
    """
    递归获取文件夹结构，包括所有子文件夹和文件
    """
    structure = []
    
    # 如果是第一层（根目录），直接处理其内容
    if level == 0:
        structure.append(os.path.basename(os.path.abspath(path)) or ".")
        
        try:
            items = sorted(os.listdir(path))
            # 过滤掉隐藏文件
            items = [item for item in items if not item.startswith('.')]
            
            for i, item in enumerate(items):
                item_path = os.path.join(path, item)
                is_last_item = i == len(items) - 1
                
                # 递归处理每个项目
                sub_structure = get_folder_structure(
                    item_path, 
                    prefix="", 
                    is_last=is_last_item,
                    level=1
                )
                structure.extend(sub_structure)
        except PermissionError:
            structure.append("    [权限不足，无法访问]")
    else:
        # 非根目录的处理
        item_name = os.path.basename(path)
        connector = "└── " if is_last else "├── "
        line = f"{prefix}{connector}{item_name}"
        
        # 如果是文件，添加文件大小信息
        if os.path.isfile(path):
            try:
                size = os.path.getsize(path)
                size_str = format_size(size)
                line += f" ({size_str})"
            except:
                pass
        
        structure.append(line)
        
        # 如果是目录，递归处理其内容
        if os.path.isdir(path):
            try:
                items = sorted(os.listdir(path))
                # 过滤掉隐藏文件
                items = [item for item in items if not item.startswith('.')]
                
                for i, item in enumerate(items):
                    item_path = os.path.join(path, item)
                    is_last_item = i == len(items) - 1
                    
                    # 计算新的前缀
                    extension = "    " if is_last else "│   "
                    new_prefix = prefix + extension
                    
                    # 递归处理
                    sub_structure = get_folder_structure(
                        item_path, 
                        prefix=new_prefix, 
                        is_last=is_last_item,
                        level=level + 1
                    )
                    structure.extend(sub_structure)
            except PermissionError:
                structure.append(f"{prefix}    [权限不足，无法访问]")
    
    return structure

def format_size(size):
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"

def save_folder_structure(output_file="folder_structure.txt", path=".", include_stats=True, show_file_sizes=True):
    """
    保存文件夹结构到txt文件
    """
    # 获取绝对路径
    abs_path = os.path.abspath(path)
    
    # 生成结构
    print(f"正在扫描文件夹: {abs_path}")
    print("注意：正在从上一级目录开始扫描...")
    structure = get_folder_structure(abs_path)
    
    # 准备输出内容
    output_lines = [
        "=" * 60,
        f"文件夹结构报告",
        f"扫描路径: {abs_path}",
        f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        "目录结构:",
        "-" * 40,
    ]
    
    # 添加结构内容
    output_lines.extend(structure)
    
    # 如果需要统计信息
    if include_stats:
        file_count = 0
        dir_count = 0
        total_size = 0
        file_types = {}
        
        for root, dirs, files in os.walk(abs_path):
            # 过滤隐藏目录
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            dir_count += len(dirs)
            
            for file in files:
                if not file.startswith('.'):
                    file_count += 1
                    # 统计文件类型
                    ext = os.path.splitext(file)[1].lower()
                    if ext:
                        file_types[ext] = file_types.get(ext, 0) + 1
                    
                    try:
                        file_path = os.path.join(root, file)
                        total_size += os.path.getsize(file_path)
                    except:
                        pass
        
        output_lines.extend([
            "",
            "-" * 40,
            "统计信息:",
            f"文件夹数量: {dir_count}",
            f"文件数量: {file_count}",
            f"总大小: {format_size(total_size)}",
        ])
        
        # 显示文件类型统计（只显示前10种）
        if file_types:
            output_lines.append("\n文件类型分布:")
            sorted_types = sorted(file_types.items(), key=lambda x: x[1], reverse=True)[:10]
            for ext, count in sorted_types:
                output_lines.append(f"  {ext}: {count} 个文件")
            if len(file_types) > 10:
                output_lines.append(f"  ... 以及其他 {len(file_types) - 10} 种类型")
        
        output_lines.append("=" * 60)
    
    # 写入文件
    output_path = os.path.abspath(output_file)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    print(f"文件夹结构已保存到: {output_path}")
    print(f"共扫描 {file_count} 个文件，{dir_count} 个文件夹")
    return file_count, dir_count

def main():
    """
    主函数
    """
    # 配置选项
    output_filename = "folder_structure.txt"  # 输出文件名（保存在当前目录）
    scan_path = "."  # 要扫描的路径（..表示上一级目录）
    include_statistics = True  # 是否包含统计信息
    
    # 也可以使用其他路径，例如：
    # scan_path = "../.."  # 上两级目录
    # scan_path = "/Users/username/Documents"  # 绝对路径
    # scan_path = "../other_folder"  # 上一级的其他文件夹
    
    # 执行扫描并保存
    file_count, dir_count = save_folder_structure(
        output_file=output_filename,
        path=scan_path,
        include_stats=include_statistics
    )
    
    # 显示预览
    print("\n预览（前30行）:")
    print("-" * 40)
    try:
        with open(output_filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines[:30]):
                print(line.rstrip())
            if len(lines) > 30:
                print(f"... (文件共 {len(lines)} 行)")
    except Exception as e:
        print(f"读取文件出错: {e}")

if __name__ == "__main__":
    main()