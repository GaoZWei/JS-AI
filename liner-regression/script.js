import * as tfvis from '@tensorflow/tfjs-vis'
import * as tf from '@tensorflow/tfjs'
window.onload = () => {
    const xs = [1, 2, 3, 4];
    const ys = [1, 3, 5, 7];

    tfvis.render.scatterplot({
        name: '线性回归样本'
    }, {
        values: xs.map((x, i) => ({
            x,
            y: ys[i]
        }))
    }, {
        xAxisDomain: [0, 5],
        yAxisDomain: [0, 8]
    });
    const model = tf.sequential() //连续模型(这一层输入是下一层输出)
    model.add(tf.layers.dense({units:1,inputShape:[1]}))//全连接层  inputShape至少为1
}