# 检查SC171环境依赖的脚本
import sys
import pkg_resources

required = [r.strip() for r in open('../requirements_sc171.txt') if r.strip() and not r.startswith('#')]
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = [r for r in required if r.split('==')[0] not in installed]

if missing:
    print('缺少依赖:', missing)
    sys.exit(1)
else:
    print('所有依赖已安装')
