import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { MnistData } from './data';

window.onload = async () => {
    const data = new MnistData();
    await data.load();
    const examples = data.nextTestBatch(20);//加载验证集
    const surface = tfvis.visor().surface({ name: '输入示例' });//加头部
    for (let i = 0; i < 20; i += 1) {
        const imageTensor = tf.tidy(() => {//tidy清除缓存(中间),防止内存泄漏
            return examples.xs
                .slice([i, 0], [1, 784]) //[第一维的起点,第二维的起点],[第二维的长度,第二维的长度]
                .reshape([28, 28, 1]);//改变Tensor形状
        });

        const canvas = document.createElement('canvas');
        canvas.width = 28;
        canvas.height = 28;
        canvas.style = 'margin: 4px';
        await tf.browser.toPixels(imageTensor, canvas);//渲染tensor
        surface.drawArea.appendChild(canvas);
    }

    const model = tf.sequential();
    model.add(tf.layers.conv2d({//2维
        inputShape: [28, 28, 1],//图片尺寸 1黑白
        kernelSize: 5,//卷积核 (奇数)
        filters: 8, 
        strides: 1,//移动步长
        activation: 'relu',//max(0,x)
        kernelInitializer: 'varianceScaling'//卷积核初始化算法
    }));
    model.add(tf.layers.maxPool2d({//池化层
        poolSize: [2, 2],
        strides: [2, 2]//移动步长
    }));
    model.add(tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));
    model.add(tf.layers.maxPool2d({
        poolSize: [2, 2],
        strides: [2, 2]
    }));
    model.add(tf.layers.flatten());//2维=>1维
    model.add(tf.layers.dense({
        units: 10, //输出10个分类
        activation: 'softmax',
        kernelInitializer: 'varianceScaling'
    }));
    model.compile({
        loss: 'categoricalCrossentropy',//交叉熵
        optimizer: tf.train.adam(),
        metrics: ['accuracy']
    });

    const [trainXs, trainYs] = tf.tidy(() => {
        const d = data.nextTrainBatch(1000);
        return [
            d.xs.reshape([1000, 28, 28, 1]),
            d.labels
        ];
    });

    const [testXs, testYs] = tf.tidy(() => {
        const d = data.nextTestBatch(200);
        return [
            d.xs.reshape([200, 28, 28, 1]),
            d.labels
        ];
    });

    await model.fit(trainXs, trainYs, {
        validationData: [testXs, testYs],
        batchSize: 500,
        epochs: 20,
        callbacks: tfvis.show.fitCallbacks(
            { name: '训练效果' },
            ['loss', 'val_loss', 'acc', 'val_acc'],
            { callbacks: ['onEpochEnd'] }
        )
    });

    const canvas = document.querySelector('canvas');

    canvas.addEventListener('mousemove', (e) => {
        if (e.buttons === 1) {
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = 'rgb(255,255,255)';
            ctx.fillRect(e.offsetX, e.offsetY, 25, 25);
        }
    });

    window.clear = () => {//清除
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = 'rgb(0,0,0)';
        ctx.fillRect(0, 0, 300, 300);//画矩形
    };

    clear();

    window.predict = () => {
        const input = tf.tidy(() => {
            return tf.image.resizeBilinear(
                tf.browser.fromPixels(canvas),// to tensor
                [28, 28],
                true
            ).slice([0, 0, 0], [28, 28, 1])//截取黑白:1
            .toFloat()
            .div(255)//归一化
            .reshape([1, 28, 28, 1]);
        });
        const pred = model.predict(input).argMax(1);
        alert(`预测结果为 ${pred.dataSync()[0]}`);
    };
};