import subprocess

# 读取 requirements.txt 文件中的包
with open("requirements.txt", "r") as file:
    packages = [line.strip() for line in file.readlines() if line.strip()]

# 获取已安装的包
installed_packages = subprocess.check_output(["pip", "list"]).decode("utf-8").splitlines()
installed_packages = [pkg.split()[0] for pkg in installed_packages[2:]]  # 排除标题

# 保存未安装的包
not_installed_packages = []

# 安装未安装的包
for package in packages:
    package_name = package.split("==")[0]  # 获取包名，排除版本号
    if package_name not in installed_packages:
        print(f"Installing {package}...")
        subprocess.call(["pip", "install", package])
        not_installed_packages.append(package)  # 记录未安装的包
    else:
        print(f"{package_name} is already installed.")

# 列出未安装的包
if not_installed_packages:
    print("\nThe following packages were not installed and have been installed now:")
    for pkg in not_installed_packages:
        print(pkg)
else:
    print("\nAll packages were already installed.")
