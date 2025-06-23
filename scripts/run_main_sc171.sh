#!/bin/bash
# 启动SC171主程序
cd /home/fibo/fiboaisdk_ubuntu_aarch64
. ./scripts/env_qualcomm.sh 68
cd /home/fibo/sc171_monitor_project
python3 ../main_sc171.py
