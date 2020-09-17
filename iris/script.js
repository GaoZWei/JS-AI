import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { getIrisData, IRIS_CLASSES } from './data';

window.onload = async () => {
    const [xTrain, yTrain, xTest, yTest] = getIrisData(0.15);//0.15作为验证集比例

    const model = tf.sequential();
    model.add(tf.layers.dense({
        units: 10,
        inputShape: [xTrain.shape[1]],
        activation: 'sigmoid'//非线性变化sigmoid
    }));
    model.add(tf.layers.dense({
        units: 3,   //3类,3个概率
        activation: 'softmax'//3个和为1 3个概率
    }));//多分类神经网络核心

    model.compile({
        loss: 'categoricalCrossentropy',//交叉熵损失函数(对数损失函数的多分类版本)
        optimizer: tf.train.adam(0.1),
        metrics: ['accuracy']//度量
    });

    await model.fit(xTrain, yTrain, {
        epochs: 100,
        validationData: [xTest, yTest],//验证集
        callbacks: tfvis.show.fitCallbacks(
            { name: '训练效果' },
            ['loss', 'val_loss', 'acc', 'val_acc'],
            { callbacks: ['onEpochEnd'] }
        )
    });

    window.predict = (form) => {
        const input = tf.tensor([[
            form.a.value * 1,
            form.b.value * 1,
            form.c.value * 1,
            form.d.value * 1,
        ]]);
        const pred = model.predict(input);
        alert(`预测结果：${IRIS_CLASSES[pred.argMax(1).dataSync(0)]}`);
    };
};