# Lightop

## 安装

环境依赖：
pytorch 1.10
dtk22.04.2

配置环境变量：
export CPLUS_INCLUDE_PATH=$ROCM_PATH/include:$ROCM_PATH/hiprand/include:$ROCM_PATH/rocrand/include:$CPLUS_INCLUDE_PATH

运行安装脚本：
python3 setup.py install

## 测试
进入test文件夹
运行任意一个测试脚本，例如：
python3 test_ln_add_dropout.py

测试模型时，参考测试脚本里调用op的方法。
