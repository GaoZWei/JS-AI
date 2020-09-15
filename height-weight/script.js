import * as tfvis from '@tensorflow/tfjs-vis'
import * as tf from '@tensorflow/tfjs'
window.onload = async() => {
    const heights = [150, 160, 170]
    const weights = [40, 50, 60]

    tfvis.render.scatterplot({
            name: '身高体重训练数据'
        }, {
            values: heights.map((x, i) => ({
                x,
                y: weights[i]
            }))
        }, {
            xAxisDomain: [140, 180],
            yAxisDomain: [30, 70]
        }
    );
    //归一化
    const inputs=tf.tensor(heights).sub(150).div(20)
    const labels=tf.tensor(weights).sub(40).div(20)

    //构建神经网络
    const model = tf.sequential() //连续模型(这一层输入是下一层输出)
    model.add(tf.layers.dense({units:1,inputShape:[1]}))//全连接层  inputShape至少为1
    //损失函数+优化
    model.compile({loss:tf.losses.meanSquaredError,optimizer:tf.train.sgd(0.1)})//损失函数中的均方误差    随机梯度下降法=>sgd学习率
   
    await model.fit(inputs,labels,{
        batchSize:3,//小批量
        epochs:100,//迭代次数
        callbacks:tfvis.show.fitCallbacks(
            {name:"训练过程"},['loss'])
    })

    //预测
    const output=model.predict(tf.tensor([180]).sub(150).div(20));
    // 反归一化
    console.log('身高180cm,体重为'+output.mul(20).add(40).dataSync()[0]);
}