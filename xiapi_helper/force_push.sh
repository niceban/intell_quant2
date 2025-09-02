#!/bin/bash
# 这是一个在当前分支直接强制覆盖main分支的脚本

# 添加所有更改
git add .

# 提交更改
echo "请输入提交信息(或直接回车使用默认信息):"
read commit_message
if [ -z "$commit_message" ]; then
  commit_message="强制更新提交"
fi
git commit -m "$commit_message"

git push -f origin HEAD:main
echo "已成功强制推送到远程main分支"