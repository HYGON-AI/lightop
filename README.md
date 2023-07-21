# Lightop

## 安装

环境依赖：
pytorch 1.10, pytorch 1.13
dtk22.04.2, dtk22.10, dtk23.04

运行安装脚本：
python3 setup.py install

## 测试
进入test文件夹
运行任意一个测试脚本，例如：
python3 test_ln_add_dropout.py

测试模型时，参考测试脚本里调用op的方法。

## 算子简介

以下时间均为单次前向传播和反向传播的总时间，包括了CPU时间和GPU时间。

<table>
  <th>算子</th><th>测试size</th><th>优化前</th><th>优化后</th><th>性能提升</th><th>使用说明</th>
    <tr>
        <td>droppath</td><td>128,3136,512</td><td>8.50ms</td><td>7.39ms</td><td>15.0%</td><td>Swin-Transformer</td>
    </tr>
    <tr>
        <td>ln_droppath</td><td>128,3136,512</td><td>51.80ms</td><td>25.99ms</td><td>99.3%</td><td>Swin-Transformer</td>
    </tr>
    <tr>
        <td>layernorm</td><td>128,50,768</td><td>0.89ms</td><td>0.62ms</td><td>43.5%</td><td>与torch原生的layernorm用法相同，只进行了性能优化。用于含有transformer结构的模型，如gpt,bert等</td>
    </tr>
    <tr>
        <td>add_dropout</td><td>128,50,768</td><td>0.38ms</td><td>0.44ms</td><td>15.7%</td><td>含有transformer结构的模型，如gpt,bert等</td>
    </tr>
    <tr>
        <td>ln_add_dropout</td><td>128,50,768</td><td>1.24ms</td><td>0.87ms</td><td>42.5%</td><td>含有transformer结构的模型，如gpt,bert等</td>
    </tr>
    <tr>
        <td>bn_act/bn_relu</td><td>4,10,128,128</td><td>0.56ms</td><td>0.33ms</td><td>69.7%</td><td>常见的CNN网络，如resnet，都会使用BN。act支持relu,softrelu,abs,power,clipped_relu,leaky_relu,silu_sigmoid,tanh等激活函数。bn_relu是bn_act的特例，减少一个参数，调用更方便。对于bn 2d，推荐NCHW维度中N*H*W>20000，此算子对于小size性能较差。</td>
    </tr>
    <tr>
        <td>linear_bias</td><td>1440,2048,5760</td><td>18.37ms</td><td>10.20ms</td><td>80.1%</td><td>size是mnk的值，mk*kn+bias.算子特性是size越大加速越不明显。</td>
    </tr>
    <tr>
        <td>softmax</td><td>512,4096</td><td>0.23ms</td><td>0.16ms</td><td>43.8%</td><td>一般分类模型的结尾会有softmax层。推荐size:第一维度大于300，第二维度大于等于1024小于等于4096。</td>
    </tr>
    <tr>
        <td>loss_ctc</td><td>120,60,240</td><td>4.51ms</td><td>1.14ms</td><td>396%</td><td>语音文本等RNN网络的输出层</td>
    </tr>
    <tr>
        <td>loss_rnnt</td><td>10,30,21,4000</td><td>16.52ms</td><td>2.93ms</td><td>429%</td><td>语音文本等RNN网络的输出层</td>
    </tr>

</table>
