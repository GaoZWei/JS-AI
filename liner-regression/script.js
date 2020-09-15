import * as tfvis from '@tensorflow/tfjs-vis'
import * as tf from '@tensorflow/tfjs'
window.onload = async () => {
    const xs = [1, 2, 3, 4];
    const ys = [1, 3, 5, 7];
    //可视化数据
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
    //构建神经网络
    const model = tf.sequential() //连续模型(这一层输入是下一层输出)
    model.add(tf.layers.dense({units:1,inputShape:[1]}))//全连接层  inputShape至少为1
    //损失函数+优化
    model.compile({loss:tf.losses.meanSquaredError,optimizer:tf.train.sgd(0.1)})//损失函数中的均方误差    随机梯度下降法=>sgd学习率
   
    //训练模型并可视化训练过程
    const inputs=tf.tensor(xs);
    const labels=tf.tensor(ys);
    await model.fit(inputs,labels,{
        batchSize:4,//小批量
        epochs:100,//迭代次数
        callbacks:tfvis.show.fitCallbacks(
            {name:"训练过程"},['loss'])
    })

    //预测
    const output=model.predict(tf.tensor([5]));
    // output.print()
    console.log(output.dataSync()[0]);
}