import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { getData } from './data.js';

window.onload = async () => {
    const data = getData(400);

    tfvis.render.scatterplot(
        { name: '逻辑回归训练数据' },
        {
            values: [
                data.filter(p => p.label === 1),//筛选
                data.filter(p => p.label === 0),
            ]
        }
    );

    const model = tf.sequential();
    model.add(tf.layers.dense({
        units: 1,//输出一个概率
        inputShape: [2],   //2个特征,长度为2的一维数组
        activation: 'sigmoid'  //输出在0-1之间
    }));
    model.compile({
        loss: tf.losses.logLoss,//对数损失函数
        optimizer: tf.train.adam(0.1)//优化器 学习率
    });

    const inputs = tf.tensor(data.map(p => [p.x, p.y]));//重点
    const labels = tf.tensor(data.map(p => p.label));

    await model.fit(inputs, labels, {
        batchSize: 40,//批量
        epochs: 20, //迭代次数
        callbacks: tfvis.show.fitCallbacks(
            { name: '训练效果' },
            ['loss']
        )
    });

    window.predict = (form) => {
        const pred = model.predict(tf.tensor([[form.x.value * 1, form.y.value * 1]]));
        alert(`预测结果：${pred.dataSync()[0]}`);
    };
};