import * as speechCommands from '@tensorflow-models/speech-commands';
import * as tfvis from '@tensorflow/tfjs-vis';

const MODEL_PATH = 'http://127.0.0.1:8080';
let transferRecognizer;//迁移学习器

window.onload = async () => {//创建学习器
    const recognizer = speechCommands.create(
        'BROWSER_FFT',
        null,
        MODEL_PATH + '/speech/model.json',
        MODEL_PATH + '/speech/metadata.json'
    );
    await recognizer.ensureModelLoaded();
    transferRecognizer = recognizer.createTransfer('轮播图');
};

window.collect = async (btn) => {//收集声音
    btn.disabled = true;
    const label = btn.innerText;
    await transferRecognizer.collectExample(
        label === '背景噪音' ? '_background_noise_' : label
    );
    btn.disabled = false;
    document.querySelector('#count').innerHTML = JSON.stringify(transferRecognizer.countExamples(), null, 2);
};

window.train = async () => {//训练
    await transferRecognizer.train({
        epochs: 30,
        callback: tfvis.show.fitCallbacks(
            { name: '训练效果' },
            ['loss', 'acc'],
            { callbacks: ['onEpochEnd'] }
        )
    });
};

window.toggle = async (checked) => {//监听
    if (checked) {
        await transferRecognizer.listen(result => {
            const { scores } = result;
            const labels = transferRecognizer.wordLabels();
            const index = scores.indexOf(Math.max(...scores));//选取最大的可能的
            console.log(labels[index]);
        }, {
            overlapFactor: 0,//识别频率
            probabilityThreshold: 0.75//准确度
        });
    } else {
        transferRecognizer.stopListening();
    }
};

window.save = () => {
    const arrayBuffer = transferRecognizer.serializeExamples();//导出成序列数据
    const blob = new Blob([arrayBuffer]);//转换为二进制
    const link = document.createElement('a');
    link.href = window.URL.createObjectURL(blob);
    link.download = 'data.bin';//下载到data文件夹的slider中bin文件
    link.click();
};