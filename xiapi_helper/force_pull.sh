#!/bin/bash

# 强制 git pull 脚本（支持分离头指针状态）

# 检查当前目录是否是 git 仓库
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
  echo "错误：当前目录不是 git 仓库"
  exit 1
fi

# 获取默认分支名称（兼容 main/master）
default_branch=$(git remote show origin | awk '/HEAD branch/ {print $3}')

# 获取当前分支名（兼容分离头指针状态）
if current_branch=$(git symbolic-ref --short HEAD 2>/dev/null); then
  echo "当前分支: $current_branch"
else
  echo "警告：当前处于分离头指针状态，将强制重置默认分支 $default_branch"
  current_branch="$default_branch"
  git checkout -f "$default_branch"  # 强制切换到默认分支
fi

echo "准备强制拉取远程更新..."

# 清理本地修改
git reset --hard

# 同步远程信息
git fetch --all

# 强制对齐远程分支
git reset --hard "origin/$current_branch"

echo "已成功强制拉取远程更新!"