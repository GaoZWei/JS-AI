import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { getData } from './data.js';

window.onload = async () => {
    const data = getData(400);

    tfvis.render.scatterplot(
        { name: 'XOR 训练数据' },
        {
            values: [
                data.filter(p => p.label === 1),
                data.filter(p => p.label === 0),
            ]
        }
    );

    const model = tf.sequential();//初始化模型
    model.add(tf.layers.dense({//隐藏层
        units: 4,//第一层4比较稳定
        inputShape: [2],
        activation: 'relu'//必须有!!!否则只能线性
    }));
    model.add(tf.layers.dense({//输出层
        units: 1,//输出一个概率
        activation: 'sigmoid'//输出一个概率,只能sigmoid
    }));
    model.compile({
        loss: tf.losses.logLoss,//本质上逻辑回归
        optimizer: tf.train.adam(0.1)
    });

    const inputs = tf.tensor(data.map(p => [p.x, p.y]));
    const labels = tf.tensor(data.map(p => p.label));

    await model.fit(inputs, labels, {
        epochs: 10,
        callbacks: tfvis.show.fitCallbacks(//可视化训练过程
            { name: '训练效果' },
            ['loss']
        )
    });

    window.predict = (form) => {
        const pred = model.predict(tf.tensor([[form.x.value * 1, form.y.value * 1]]));
        alert(`预测结果：${pred.dataSync()[0]}`);
    };
};