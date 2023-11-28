# Lightop

## 安装

环境依赖：
pytorch1.10 \ pytorch1.13 \ pytorch2.0 \ pytorch2.1

dtk22.04.2 \ dtk22.10 \ dtk23.04 \ dtk23.10

安装脚本：

python3 setup.py install

## 测试
进入test文件夹

运行任意一个测试脚本，例如：

python3 test_ln_add_dropout.py

运行python3 all_test.py可以测试全部脚本

测试模型时，参考测试脚本里调用算子的方法。

## 算子简介

以下时间均为单次前向传播和反向传播的总时间，包括了CPU时间和GPU时间。测试环境：pytorch1.13, dtk23.04, 10.6.10.67服务器z100计算卡。

<table>
  <th>算子</th><th>测试size</th><th>优化前</th><th>优化后</th><th>性能提升</th><th>使用说明</th>
    <tr>
        <td>droppath</td><td>128,3136,96</td><td>0.84ms</td><td>0.74ms</td><td>13.5%</td><td>Swin-Transformer</td>
    </tr>
    <tr>
        <td>ln_droppath</td><td>128, 3136, 96</td><td>15.7ms</td><td>4.06ms</td><td>287%</td><td>Swin-Transformer. droppath(ln(input))</td>
    </tr>
    <tr>
        <td>layernorm</td><td>1, 2048, 3072</td><td>0.58ms</td><td>0.32ms</td><td>81.3%</td><td>NLP类</td>
    </tr>
    <tr>
        <td>ln_dropout</td><td>1, 2048, 3072</td><td>0.78ms</td><td>0.42ms</td><td>85.7%</td><td>NLP类.ln(dropout(input))</td>
    </tr>
    <tr>
        <td>ln_add_dropout</td><td>1, 2048, 3072</td><td>0.97ms</td><td>0.47ms</td><td>106%</td><td>NLP类.ln(dropout(input)+residual)</td>
    </tr>
    <tr>
        <td>ln_add_dropout_bias</td><td>1, 2048, 3072</td><td>1.16ms</td><td>0.53ms</td><td>119%</td><td>NLP类.ln(dropout(input+bias)+residual)</td>
    </tr>
    <tr>
        <td>rmsnorm</td><td>1, 2048, 3072</td><td>1.28ms</td><td>0.29ms</td><td>341%</td><td>NLP类.</td>
    </tr>
    <tr>
        <td>rn_dropout</td><td>1, 2048, 3072</td><td>1.51ms</td><td>0.38ms</td><td>297%</td><td>NLP类.rn(dropout(input))</td>
    </tr>
    <tr>
        <td>rn_add_dropout</td><td>1, 2048, 3072</td><td>1.68ms</td><td>0.45ms</td><td>273%</td><td>NLP类.rn(dropout(input)+residual)</td>
    </tr>
    <tr>
        <td>rn_add_dropout_bias</td><td>1, 2048, 3072</td><td>1.88ms</td><td>0.51ms</td><td>268%</td><td>NLP类.rn(dropout(input+bias)+residual)</td>
    </tr>
    <tr>
        <td>add_dropout</td><td>1, 2048, 3072</td><td>0.49ms</td><td>0.38ms</td><td>28.9%</td><td>NLP类.dropout(input)+residual</td>
    </tr>
    <tr>
        <td>bias_dropout_add</td><td>1, 2048, 3072</td><td>0.69ms</td><td>0.43ms</td><td>60.5%</td><td>NLP类.dropout(input+bias)+residual</td>
    </tr>
    <tr>
        <td>bias_gelu</td><td>1, 2048, 3072</td><td>0.38ms</td><td>0.27ms</td><td>40.7%</td><td>NLP类,gelu(input+bias)</td>
    </tr>
    <tr>
        <td>bias_swiglu</td><td>2, 2048, 512</td><td>0.50ms</td><td>0.28ms</td><td>78.5%</td><td>NLP类SwiGLU(input+bias)</td>
    </tr>
    <tr>
        <td>FusedScaleMaskSoftmaxDropout</td><td>2,16,128,128</td><td>0.26ms</td><td>0.34ms</td><td>30.7%</td><td>NLP类.</td>
    </tr>
    <tr>
        <td>bn_act/bn_relu</td><td>4,10,128,128</td><td>0.56ms</td><td>0.33ms</td><td>69.7%</td><td>CNN，如resnet，都会使用BN。act支持relu,softrelu,abs,power,clipped_relu,leaky_relu,silu_sigmoid,tanh等激活函数。bn_relu是bn_act的特例，减少一个参数，调用更方便。对于bn 2d，推荐NCHW维度中N*H*W>20000，此算子对于小size性能较差。</td>
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
